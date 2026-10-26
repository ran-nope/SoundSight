from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_starts():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "🎧 SoundSight"
