"""Shared training/inference preprocessing. No fitted state here."""
from dataclasses import dataclass
import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    seconds: float = 5.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    n_mfcc: int = 20


def load_audio(source, config=AudioConfig()):
    """Decode bounded audio (including MP3); mix, resample, pad/crop to 5 s."""
    with sf.SoundFile(source) as audio:
        if audio.frames == 0 or audio.samplerate <= 0:
            raise ValueError("Audio is empty.")
        if audio.frames / audio.samplerate > 30:
            raise ValueError("Please use a clip of 30 seconds or less.")
        sr = audio.samplerate
        y = audio.read(dtype="float32", always_2d=True).mean(axis=1)
    if not np.isfinite(y).all():
        raise ValueError("Audio contains invalid samples.")
    y = librosa.resample(y, orig_sr=sr, target_sr=config.sample_rate)
    y = librosa.util.fix_length(y, size=int(config.seconds * config.sample_rate))
    if np.max(np.abs(y)) < 1e-7:
        raise ValueError("Audio is silent; choose an audible recording.")
    return y


def features(y, config=AudioConfig()):
    power = np.abs(librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)) ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=config.sample_rate, n_mels=config.n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=config.n_mfcc)
    blocks = [logmel, mfcc, librosa.feature.delta(mfcc),
              librosa.feature.spectral_centroid(S=np.sqrt(power), sr=config.sample_rate),
              librosa.feature.spectral_bandwidth(S=np.sqrt(power), sr=config.sample_rate),
              librosa.feature.spectral_rolloff(S=np.sqrt(power), sr=config.sample_rate),
              librosa.feature.zero_crossing_rate(y, frame_length=config.n_fft, hop_length=config.hop_length),
              librosa.feature.rms(S=np.sqrt(power), frame_length=config.n_fft)]
    vector = np.concatenate([stat(b, axis=1) for b in blocks for stat in (np.mean, np.std)])
    if not np.isfinite(vector).all():
        raise ValueError("Could not extract finite audio features.")
    return vector.astype(np.float32)


def extract(source, config=AudioConfig()):
    return features(load_audio(source, config), config)
