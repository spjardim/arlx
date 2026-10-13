import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from arlx.audio.sources.base import AudioSource
from arlx.audio.sources.input_capture import InputAudioCapture

FAKE_DEVICE = {"name": "Fake Device", "default_samplerate": 48000, "max_input_channels": 2}


def make_status(input_overflow=False):
    return SimpleNamespace(input_overflow=input_overflow)


def make_time_info(adc_time=1.0):
    return SimpleNamespace(inputBufferAdcTime=adc_time)


def test_input_audio_capture_is_an_audio_source():
    assert issubclass(InputAudioCapture, AudioSource)


def test_set_device_name_updates_attribute():
    capture = InputAudioCapture(device_name="default")
    capture.set_device_name("other")
    assert capture.device_name == "other"


# ---- callback / sample retrieval (pure logic, no sounddevice needed) ----

def test_callback_stores_latest_sample_and_timestamp():
    capture = InputAudioCapture()
    indata = np.array([[1.0], [2.0]], dtype=np.float32)

    capture.callback(indata, frames=2, time_info=make_time_info(5.0), status=make_status())

    assert np.array_equal(capture.get_latest_sample(), indata)
    assert capture.latest_timestamp == 5.0


def test_callback_copies_indata_so_later_mutation_does_not_affect_stored_sample():
    capture = InputAudioCapture()
    indata = np.array([[1.0]], dtype=np.float32)

    capture.callback(indata, frames=1, time_info=make_time_info(), status=make_status())
    indata[0, 0] = 999.0

    assert capture.get_latest_sample()[0, 0] == 1.0


def test_callback_logs_warning_on_input_overflow(caplog):
    capture = InputAudioCapture()
    indata = np.zeros((1, 1), dtype=np.float32)

    with caplog.at_level(logging.WARNING):
        capture.callback(indata, frames=1, time_info=make_time_info(), status=make_status(input_overflow=True))

    assert "overflow" in caplog.text


def test_get_latest_sample_with_timestamp_before_any_callback_returns_true_with_no_data():
    capture = InputAudioCapture()
    returned, sample, timestamp = capture.get_latest_sample_with_timestamp()
    assert returned is True
    assert sample is None
    assert timestamp is None


def test_get_latest_sample_with_timestamp_returns_true_once_for_a_new_sample():
    capture = InputAudioCapture()
    capture.callback(np.zeros((1, 1)), frames=1, time_info=make_time_info(1.0), status=make_status())

    returned, sample, timestamp = capture.get_latest_sample_with_timestamp()
    assert returned is True
    assert timestamp == 1.0

    returned_again, sample_again, timestamp_again = capture.get_latest_sample_with_timestamp()
    assert returned_again is False
    assert sample_again is None
    assert timestamp_again is None


def test_get_latest_sample_with_timestamp_returns_true_again_after_a_newer_sample():
    capture = InputAudioCapture()
    capture.callback(np.zeros((1, 1)), frames=1, time_info=make_time_info(1.0), status=make_status())
    capture.get_latest_sample_with_timestamp()

    capture.callback(np.ones((1, 1)), frames=1, time_info=make_time_info(2.0), status=make_status())
    returned, sample, timestamp = capture.get_latest_sample_with_timestamp()
    assert returned is True
    assert timestamp == 2.0


# ---- start / stop, with sounddevice mocked out so no real hardware is touched ----

@patch("arlx.audio.sources.input_capture.sd")
def test_start_success_configures_and_starts_an_input_stream(mock_sd):
    mock_sd.query_devices.return_value = FAKE_DEVICE
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    capture = InputAudioCapture(device_name="default", block_size=1024)
    result = capture.start()

    assert result is True
    assert capture.stream_connected is True
    mock_sd.InputStream.assert_called_once_with(
        device="Fake Device",
        channels=1,
        samplerate=48000,
        blocksize=1024,
        callback=capture.callback,
    )
    mock_stream.start.assert_called_once()


@patch("arlx.audio.sources.input_capture.sd")
def test_start_failure_leaves_stream_disconnected(mock_sd):
    mock_sd.query_devices.side_effect = Exception("no such device")

    capture = InputAudioCapture(device_name="missing")
    result = capture.start()

    assert result is False
    assert capture.stream_connected is False


@patch("arlx.audio.sources.input_capture.sd")
def test_stop_after_successful_start_closes_the_stream(mock_sd):
    mock_sd.query_devices.return_value = FAKE_DEVICE
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    capture = InputAudioCapture()
    capture.start()
    result = capture.stop()

    assert result is True
    assert capture.stream_connected is False
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()


def test_stop_without_ever_starting_is_a_harmless_no_op():
    capture = InputAudioCapture()
    assert capture.stop() is True


@patch("arlx.audio.sources.input_capture.sd")
def test_stop_failure_is_caught_and_reported(mock_sd):
    mock_sd.query_devices.return_value = FAKE_DEVICE
    mock_stream = MagicMock()
    mock_stream.stop.side_effect = Exception("device unplugged")
    mock_sd.InputStream.return_value = mock_stream

    capture = InputAudioCapture()
    capture.start()
    result = capture.stop()

    assert result is False
    assert capture.stream_connected is True


@patch("arlx.audio.sources.input_capture.sd")
def test_list_potential_devices_filters_out_devices_with_zero_input_channels(mock_sd):
    mock_sd.query_devices.return_value = [
        {"name": "Mic", "max_input_channels": 2},
        {"name": "Speakers", "max_input_channels": 0},
        {"name": "Interface", "max_input_channels": 1},
    ]

    capture = InputAudioCapture()
    devices = capture.list_potential_input_devices()

    assert [device["name"] for device in devices] == ["Mic", "Interface"]
