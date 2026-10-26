from io import BytesIO
import joblib
import numpy as np
import pytest
import soundfile as sf
from soundsight.audio import AudioConfig, extract, load_audio
from soundsight.model import make_model, split_fold, predict, predict_audio, load_bundle


def wav(freq=440, sr=44100, seconds=1, stereo=False):
    y = 0.3 * np.sin(2 * np.pi * freq * np.arange(int(sr * seconds)) / sr)
    if stereo:
        y = np.column_stack([y, y])
    f = BytesIO()
    sf.write(f, y, sr, format="WAV")
    f.seek(0)
    return f


def test_audio_contract():
    y = load_audio(wav(stereo=True))
    assert y.shape == (110250,)
    assert np.count_nonzero(y[22060:]) == 0
    assert np.allclose(extract(wav()), extract(wav(stereo=True)), atol=0.02)
    assert extract(wav()).shape == (218,)


def test_crop_and_invalid_audio():
    assert load_audio(wav(seconds=6)).shape == (110250,)
    with pytest.raises(ValueError, match="silent"):
        load_audio(wav(freq=0))
    with pytest.raises(ValueError, match="30 seconds"):
        load_audio(wav(seconds=31))
    with pytest.raises((RuntimeError, ValueError)):
        load_audio(BytesIO(b"not audio"))


@pytest.mark.parametrize("format", ["MP3", "FLAC", "OGG", "AIFF"])
def test_audio_formats(format):
    # Encode genuine files, then run the same decoder used by the uploader.
    original, sr = sf.read(wav())
    encoded = BytesIO()
    sf.write(encoded, original, sr, format=format)
    encoded.seek(0)
    y = load_audio(encoded)
    assert y.shape == (110250,)
    assert np.isfinite(y).all()
    # Lossy encoding can change samples, but should preserve the test tone.
    spectrum = np.abs(np.fft.rfft(y[:22050]))
    frequency = np.fft.rfftfreq(22050, 1 / 22050)[spectrum.argmax()]
    assert abs(frequency - 440) < 5


def test_fold_isolation_and_training_scaler():
    folds = np.array([1, 1, 2, 2, 5, 5])
    tr, te = split_fold(folds, 5)
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [999, 999], [888, 888]])
    model = make_model().fit(x[tr], [0, 1, 0, 1])
    assert not set(tr) & set(te)
    assert np.allclose(model[0].mean_, x[tr].mean(axis=0))
    with pytest.raises(ValueError):
        split_fold(folds, 4)


def test_shared_source_is_purged_from_training():
    tr, te = split_fold([1, 2, 3, 5, 5], 5, ["a", "b", "c", "a", "d"])
    assert tr.tolist() == [1, 2]
    assert te.tolist() == [3, 4]


def test_real_fit_serialization_and_prediction(tmp_path):
    # Synthetic tones test plumbing only; this is NOT ESC-50 evaluation.
    freqs = [220, 240, 440, 460, 880, 900]
    x = np.array([extract(wav(f)) for f in freqs])
    model = make_model().fit(x, [0, 0, 1, 1, 2, 2])
    bundle = {"format_version": 1, "model": model, "audio_config": AudioConfig().__dict__, "labels": {0: "low", 1: "mid", 2: "high"}}
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    result = predict(wav(440), load_bundle(path), top_k=3)
    assert result[0]["sound"] == "mid"
    assert predict_audio(load_audio(wav(440)), load_bundle(path), top_k=3) == result
    assert all(np.isfinite(r["score"]) for r in result)
    mp3 = BytesIO()
    samples, sr = sf.read(wav(440))
    sf.write(mp3, samples, sr, format="MP3")
    mp3.seek(0)
    # Lossy encoding changes features; test successful inference, not identical labels.
    compressed_result = predict(mp3, load_bundle(path))
    assert {r["sound"] for r in compressed_result} == {"low", "mid", "high"}
    assert all(np.isfinite(r["score"]) for r in compressed_result)
