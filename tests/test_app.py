from pathlib import Path
import json
import pytest
from io import BytesIO
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from soundsight.audio import load_audio


def test_app_starts():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "SoundSight"
    assert app.session_state["source"] == "Examples"
    assert len(app.get("audio_input")) == 0
    assert len(app.get("imgs")) == 2


@pytest.mark.parametrize("index", [0, 1, 2])
def test_sample_analysis_and_upload_switch(index):
    root = Path(__file__).parents[1]
    samples = json.loads((root / "samples/manifest.json").read_text(encoding="utf-8"))
    app = AppTest.from_file(str(root / "app.py")).run(timeout=60)
    # Streamlit 1.45's ButtonGroup test adapter expects a list even for single selection.
    app.get("button_group")[0].set_value(["Examples"])
    label = samples[index]["name"].split(" ", 1)[-1]
    app.get("button_group")[1].set_value([label])
    app.run(timeout=60)
    assert not app.exception
    assert not app.error
    assert any(s.value == "Inside the recording" for s in app.subheader)
    assert any(samples[index]["creator"] in c.value for c in app.caption)
    assert len(app.get("imgs")) == 2
    app.get("button_group")[0].set_value(["Upload"])
    app.get("button_group")[1].set_value([label])
    app.run(timeout=60)
    assert not app.exception
    assert not any(s.value == "Inside the recording" for s in app.subheader)


def test_recording_reaches_prediction_without_accessing_a_microphone():
    root = Path(__file__).parents[1]
    class Recording(BytesIO):
        @property
        def size(self):
            return len(self.getvalue())
    audio = Recording((root / "samples/5-181766-A-10.wav").read_bytes())
    with patch("streamlit.audio_input", return_value=audio):
        app = AppTest.from_file(str(root / "app.py"))
        app.session_state["source"] = "Record"
        app.run(timeout=60)
    assert not app.exception
    assert not app.error
    assert any(s.value == "Inside the recording" for s in app.subheader)
    assert len(app.get("imgs")) == 2


def test_default_example_decodes_only_once():
    root = Path(__file__).parents[1]
    with patch("soundsight.audio.load_audio", wraps=load_audio) as decoder:
        app = AppTest.from_file(str(root / "app.py")).run(timeout=60)
    assert not app.exception
    assert decoder.call_count == 1
