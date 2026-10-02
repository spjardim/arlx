from arlx.audio.sources.base import AudioSource
from arlx.audio.sources.input_capture import InputAudioCapture


def test_input_audio_capture_is_an_audio_source():
    assert issubclass(InputAudioCapture, AudioSource)


def test_list_potential_devices_returns_input_capable_devices():
    audio = InputAudioCapture()
    devices = audio.list_potential_devices()
    assert all(device["max_input_channels"] > 0 for device in devices)
