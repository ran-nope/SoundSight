from pathlib import Path
import json
import pytest
from io import BytesIO
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


def test_app_starts():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "SoundSight"
    assert app.radio[0].options == ["Record", "Upload", "Examples"]
    assert len(app.get("audio_input")) == 1


@pytest.mark.parametrize("index", [0, 1, 2])
def test_sample_analysis_and_upload_switch(index):
    root = Path(__file__).parents[1]
    samples = json.loads((root / "samples/manifest.json").read_text(encoding="utf-8"))
    app = AppTest.from_file(str(root / "app.py")).run(timeout=60)
    app.radio[0].set_value("Examples").run(timeout=60)
    app.button[index].click().run(timeout=60)
    assert not app.exception
    assert not app.error
    assert any(s.value.startswith("Closest match:") for s in app.subheader)
    assert any(samples[index]["creator"] in c.value for c in app.caption)
    assert len(app.get("imgs")) == 2
    app.radio[0].set_value("Upload").run(timeout=60)
    assert not app.exception
    assert not any(s.value.startswith("Closest match:") for s in app.subheader)


def test_recording_reaches_prediction_without_accessing_a_microphone():
    root = Path(__file__).parents[1]
    class Recording(BytesIO):
        @property
        def size(self):
            return len(self.getvalue())
    audio = Recording((root / "samples/5-181766-A-10.wav").read_bytes())
    with patch("streamlit.audio_input", return_value=audio):
        app = AppTest.from_file(str(root / "app.py")).run(timeout=60)
    assert not app.exception
    assert not app.error
    assert any(s.value.startswith("Closest match:") for s in app.subheader)
    assert len(app.get("imgs")) == 2
