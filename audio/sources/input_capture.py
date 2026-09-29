import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from base import AudioSource

log = logging.getLogger(__name__)


class InputAudioCapture(AudioSource):

    def __init__(self, device_name = "default", block_size = 1024):
        self.device_name = device_name
        self.block_size = block_size

        self.stream: Optional[sd.InputStream] = None

        self.stream_connected = False

        self._lock = threading.Lock()
        self._latest_sample: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None

    def callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            log.warning("audio input overflow, samples were dropped")

        with self._lock:
            self._latest_sample = indata.copy()
            self._latest_timestamp = time_info.inputBufferAdcTime

    def list_potential_devices(self) -> list:
        return [device for device in sd.query_devices() if device["max_input_channels"] != 0]

    def set_device_name(self, device_name) -> None:
        self.device_name = device_name

    def get_latest_sample(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_sample

    def get_latest_sample_with_timestamp(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            return self._latest_sample, self._latest_timestamp

    def start(self) -> bool:
        try:
            device = sd.query_devices(self.device_name)
            self.stream = sd.InputStream(
                device=device["name"],
                channels=1,
                samplerate=device["default_samplerate"],
                blocksize=self.block_size,
                callback=self.callback,
            )
            self.stream.start()
            self.stream_connected = True
        except Exception:
            log.exception("failed to start audio stream (device=%r)", self.device_name)
            self.stream_connected = False
        return self.stream_connected

    def stop(self) -> bool:
        try:
            if self.stream_connected and self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream_connected = False
            return True
        except Exception:
            log.exception("failed to stop audio stream")
            return False
