"""Run from repository root: streamlit run app.py."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from soundsight.audio import AudioConfig, load_audio
from soundsight.model import load_bundle, predict

ROOT = Path(__file__).parent
st.set_page_config(page_title="SoundSight", page_icon="🎧", layout="wide")
st.title("🎧 SoundSight")
st.write("Explore the sounds around you. Try an example or upload your own recording.")
st.caption("A 50-class environmental sound classifier trained on ESC-50. Free hosting may take a minute to wake up.")
model_path = ROOT / "artifacts/model.joblib"
if not model_path.exists():
    st.info("Train the model first: soundsight train --cv")
    st.stop()


@st.cache_resource
def get_model(stamp):
    return load_bundle(model_path)


@st.cache_data(max_entries=12, ttl=600, show_spinner=False)
def prepare_audio(raw, settings):
    config = AudioConfig(**settings)
    y = load_audio(BytesIO(raw), config)
    mel = librosa.feature.melspectrogram(y=y, sr=config.sample_rate,
        n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)
    return y, librosa.power_to_db(mel, ref=np.max)


bundle = get_model(model_path.stat().st_mtime_ns)
st.info("The model always chooses among its 50 known classes. Unfamiliar or overlapping sounds can be misclassified. Ranking scores are not confidence percentages.")
with st.expander("Which sounds can it recognize?"):
    st.write(", ".join(label.replace("_", " ") for _, label in sorted(bundle["labels"].items())))

source = st.radio("Audio source", ["Try a sample", "Upload audio"], horizontal=True)
raw = None
if source == "Try a sample":
    samples = json.loads((ROOT / "samples/manifest.json").read_text(encoding="utf-8"))
    for column, sample in zip(st.columns(len(samples)), samples):
        if column.button(sample["name"], use_container_width=True):
            st.session_state["sample_name"] = sample["name"]
    selected = next((s for s in samples if s["name"] == st.session_state.get("sample_name")), None)
    if selected:
        raw = (ROOT / "samples" / selected["file"]).read_bytes()
        st.caption(f"Dataset label: {selected['category'].replace('_', ' ')} · ESC-50 fold 5 (held out from the demo model).")
        st.caption(f"Source: [{selected['title']}]({selected['source_url']}) by {selected['creator']} · {selected['original_license']}. ESC-10 excerpt compiled by Karol J. Piczak, CC BY 3.0.")
    else:
        st.caption("Choose a sample above to see a prediction. These are dataset examples, not an independent real-world benchmark.")
else:
    st.caption("MP3, WAV, FLAC, OGG or AIFF · up to 30 seconds / 10 MB. Only the first 5 seconds are analyzed; shorter clips are padded.")
    uploaded = st.file_uploader("Choose a sound", type=["mp3", "wav", "flac", "ogg", "aiff", "aif"])
    if uploaded is not None:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("Please upload a file smaller than 10 MB.")
        else:
            raw = uploaded.getvalue()

if raw is not None:
    try:
        with st.spinner("Listening and analyzing…"):
            y, logmel = prepare_audio(raw, bundle["audio_config"])
            results = predict(BytesIO(raw), bundle)
        st.audio(y, sample_rate=bundle["audio_config"]["sample_rate"])
        st.caption("Preview of the five-second segment used for prediction.")
        left, right = st.columns([1, 1.3])
        with left:
            st.subheader("Closest match: " + results[0]["sound"].replace("_", " ").title())
            fig, ax = plt.subplots(figsize=(6, 3.2))
            names = [r["sound"].replace("_", " ") for r in results]
            scores = [r["score"] for r in results]
            ax.barh(names, scores, color=["#0d9488"] + ["#94a3b8"] * (len(scores) - 1))
            ax.invert_yaxis()
            ax.set_xlabel("SVM ranking score (not probability)")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("Exact ranking scores"):
                st.dataframe(results, hide_index=True, use_container_width=True)
        with right:
            st.subheader("What the model heard")
            config = AudioConfig(**bundle["audio_config"])
            fig, axes = plt.subplots(2, 1, figsize=(7, 4.5))
            axes[0].plot(np.arange(len(y))[::100] / config.sample_rate, y[::100], color="#0d9488", linewidth=0.8)
            axes[0].set(xlim=(0, config.seconds), xlabel="Time (seconds)", ylabel="Amplitude", title="Waveform")
            im = librosa.display.specshow(logmel, sr=config.sample_rate, hop_length=config.hop_length,
                x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
            axes[1].set_title("Mel spectrogram")
            fig.colorbar(im, ax=axes[1], format="%+2.0f dB")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Brighter regions contain more energy. The mel scale groups frequencies to reflect how people perceive pitch.")
        st.caption("Uploads stay in the app process and its temporary in-memory cache. This app does not write uploaded audio to disk.")
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(f"Could not analyze this file: {exc}")

with st.expander("Evaluation and model details"):
    st.write(f"Demo model trained on folds {bundle['training_folds']}; fold {bundle['held_out_fold']} held out. Training clips sharing test sources were excluded.")
    metrics_path = model_path.with_name("metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("model_sha256") == hashlib.sha256(model_path.read_bytes()).hexdigest():
            st.write(metrics["protocol"])
            a, b = st.columns(2)
            a.metric("Mean held-out accuracy", f"{metrics['mean_accuracy']:.1%}")
            b.metric("Mean macro F1", f"{metrics['mean_macro_f1']:.3f}")
            st.caption("These curated-dataset results do not measure accuracy on your uploads.")
    st.markdown("[Source code and evaluation report](https://github.com/ran-nope/SoundSight)")
