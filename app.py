"""Run from repository root: streamlit run app.py."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import streamlit as st
from soundsight.audio import AudioConfig, load_audio
from soundsight.model import load_bundle, predict

st.set_page_config(page_title="SoundSight", page_icon="🎧")
st.title("🎧 SoundSight")
st.write("What does your environment sound like? Explore a 50-class audio classifier.")
st.caption("Upload MP3, WAV, FLAC, OGG or AIFF, up to 30 seconds / 10 MB. The first 5 seconds are analyzed; shorter clips are padded.")
model_path = Path(__file__).parent / "artifacts/model.joblib"
if not model_path.exists():
    st.info("Train the model first: soundsight train --cv")
    st.stop()


@st.cache_resource
def get_model(stamp):
    return load_bundle(model_path)


bundle = get_model(model_path.stat().st_mtime_ns)
st.caption(f"Demo model: trained on official folds {bundle['training_folds']}; fold {bundle['held_out_fold']} held out.")
metrics_path = model_path.with_name("metrics.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("model_sha256") == hashlib.sha256(model_path.read_bytes()).hexdigest():
        with st.expander("Measured evaluation"):
            st.write(metrics["protocol"])
            st.metric("Mean held-out accuracy", f"{metrics['mean_accuracy']:.1%}")
            st.metric("Mean macro F1", f"{metrics['mean_macro_f1']:.3f}")
            st.caption("Shared-source clips are excluded from training. These curated-dataset results do not measure performance on your uploads.")
st.warning("This demo always chooses among 50 known sounds. Mixed or unfamiliar sounds may be misclassified. Scores are rankings, not confidence probabilities.")
uploaded = st.file_uploader("Choose a sound", type=["mp3", "wav", "flac", "ogg", "aiff", "aif"])
if uploaded is not None:
    if uploaded.size > 10 * 1024 * 1024:
        st.error("Please upload a file smaller than 10 MB.")
    else:
        raw = uploaded.getvalue()
        try:
            y = load_audio(BytesIO(raw), AudioConfig(**bundle["audio_config"]))
            st.audio(y, sample_rate=bundle["audio_config"]["sample_rate"])
            st.caption("Preview of the five-second audio used for prediction.")
            st.line_chart(y[::220], x_label="Time steps (0.01 s)", y_label="Amplitude")
            results = predict(BytesIO(raw), bundle)
            st.subheader(results[0]["sound"].replace("_", " ").title())
            st.dataframe(results, hide_index=True, use_container_width=True)
            st.caption("Processing stays in the app process. Uploaded audio is not written to disk by this app.")
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Could not analyze this file: {exc}")
