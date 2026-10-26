from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest


def test_app_starts():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "🎧 SoundSight"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_sample_analysis_and_upload_switch(index):
    root = Path(__file__).parents[1]
    samples = json.loads((root / "samples/manifest.json").read_text(encoding="utf-8"))
    app = AppTest.from_file(str(root / "app.py")).run(timeout=60)
    app.button[index].click().run(timeout=60)
    assert not app.exception
    assert not app.error
    assert any(s.value.startswith("Closest match:") for s in app.subheader)
    assert any(samples[index]["creator"] in c.value for c in app.caption)
    assert len(app.get("imgs")) == 2
    app.radio[0].set_value("Upload audio").run(timeout=60)
    assert not app.exception
    assert not app.subheader
