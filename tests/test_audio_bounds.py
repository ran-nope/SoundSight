from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from soundsight.audio import load_audio


@pytest.mark.parametrize('rate,channels,frames', [(384000, 2, 100), (44100, 64, 100), (44100, 2, -1), (44100, 2, 44100*31)])
def test_reject_before_read(rate, channels, frames):
    audio = MagicMock(samplerate=rate, channels=channels, frames=frames)
    audio.__enter__.return_value = audio
    with patch('soundsight.audio.sf.SoundFile', return_value=audio):
        with pytest.raises(ValueError):
            load_audio('unused')
    audio.read.assert_not_called()


def test_only_analysis_window_is_read():
    audio = MagicMock(samplerate=22050, channels=2, frames=22050*30)
    audio.__enter__.return_value = audio
    audio.read.return_value = np.ones((22050*5, 2), dtype=np.float32)
    with patch('soundsight.audio.sf.SoundFile', return_value=audio):
        assert len(load_audio('unused')) == 22050*5
    audio.read.assert_called_once_with(frames=22050*5, dtype='float32', always_2d=True)
