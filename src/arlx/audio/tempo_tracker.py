from collections import deque

import numpy as np

DEFAULT_MIN_BPM = 70.0
DEFAULT_MAX_BPM = 180.0
DEFAULT_HISTORY_DURATION = 4.0 
DEFAULT_SMOOTHING = 0.2  


class TempoTracker:
    def __init__(self,
                 sample_rate: int = 44100,
                 block_size: int = 512,
                 min_bpm: float = DEFAULT_MIN_BPM,
                 max_bpm: float = DEFAULT_MAX_BPM,
                 history_duration: float = DEFAULT_HISTORY_DURATION,
                 smoothing: float = DEFAULT_SMOOTHING,
                 ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.history_duration = history_duration
        self.smoothing = smoothing

        self.block_duration = self.block_size / self.sample_rate
        self.history_length = max(int(self.history_duration / self.block_duration), 1)

        self.min_lag_blocks = max(1, round((60.0 / self.max_bpm) / self.block_duration))
        self.max_lag_blocks = max(self.min_lag_blocks + 1, round((60.0 / self.min_bpm) / self.block_duration))
        self.max_lag_blocks = min(self.max_lag_blocks, self.history_length - 1)

        self.onset_history = deque(maxlen=self.history_length)

        self.bpm = 0.0
        self.confidence = 0.0
        self.autocorrelation = np.zeros(self.max_lag_blocks - self.min_lag_blocks + 1)

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate

        self.block_duration = self.block_size / self.sample_rate
        self.history_length = max(int(self.history_duration / self.block_duration), 1)

        self.min_lag_blocks = max(1, round((60.0 / self.max_bpm) / self.block_duration))
        self.max_lag_blocks = max(self.min_lag_blocks + 1, round((60.0 / self.min_bpm) / self.block_duration))
        self.max_lag_blocks = min(self.max_lag_blocks, self.history_length - 1)

        self.onset_history = deque(maxlen=self.history_length)

        self.autocorrelation = np.zeros(self.max_lag_blocks - self.min_lag_blocks + 1)

    def get_block_size(self):
        return self.block_size

    def set_block_size(self, block_size: int):
        self.block_size = block_size
        self.block_duration = self.block_size / self.sample_rate
        self.history_length = max(int(self.history_duration / self.block_duration), 1)

        self.min_lag_blocks = max(1, round((60.0 / self.max_bpm) / self.block_duration))
        self.max_lag_blocks = max(self.min_lag_blocks + 1, round((60.0 / self.min_bpm) / self.block_duration))
        self.max_lag_blocks = min(self.max_lag_blocks, self.history_length - 1)

        self.onset_history = deque(maxlen=self.history_length)

        self.autocorrelation = np.zeros(self.max_lag_blocks - self.min_lag_blocks + 1)

    def get_min_bpm(self) -> float:
        return self.min_bpm

    def set_min_bpm(self, min_bpm: float):
        self.min_bpm = min_bpm
        self._recompute_lag_bounds()

    def get_max_bpm(self) -> float:
        return self.max_bpm

    def set_max_bpm(self, max_bpm: float):
        self.max_bpm = max_bpm
        self._recompute_lag_bounds()

    def get_history_duration(self) -> float:
        return self.history_duration

    def set_history_duration(self, history_duration: float):
        self.history_duration = history_duration
        self.history_length = max(int(history_duration / self.block_duration), 1)
        self.onset_history = deque(self.onset_history, maxlen=self.history_length)
        self._recompute_lag_bounds()

    def get_smoothing(self) -> float:
        return self.smoothing

    def set_smoothing(self, smoothing: float):
        self.smoothing = smoothing

    def _recompute_lag_bounds(self):
        self.min_lag_blocks = max(1, round((60.0 / self.max_bpm) / self.block_duration))
        self.max_lag_blocks = max(self.min_lag_blocks + 1, round((60.0 / self.min_bpm) / self.block_duration))
        self.max_lag_blocks = min(self.max_lag_blocks, self.history_length - 1)
        self.autocorrelation = np.zeros(self.max_lag_blocks - self.min_lag_blocks + 1)

    def get_bpm(self) -> float:
        return self.bpm

    def get_confidence(self) -> float:
        return self.confidence

    def get_autocorrelation(self) -> np.ndarray:
        return self.autocorrelation.copy()

    def get_bpm_axis(self) -> np.ndarray:
        lags = np.arange(self.min_lag_blocks, self.max_lag_blocks + 1)
        return 60.0 / (lags * self.block_duration)

    def process(self, onset: bool) -> None:
        self.onset_history.append(1.0 if onset else 0.0)

        warmed_up = len(self.onset_history) == self.history_length
        if not warmed_up:
            return

        signal = np.array(self.onset_history)
        zero_lag_energy = np.dot(signal, signal)

        if zero_lag_energy <= 0:
            self.confidence = 0.0
            self.autocorrelation = np.zeros_like(self.autocorrelation)
            return

        lags = np.arange(self.min_lag_blocks, self.max_lag_blocks + 1)
        self.autocorrelation = np.array([
            np.dot(signal[:len(signal) - lag], signal[lag:]) for lag in lags
        ])

        peak_index = np.argmax(self.autocorrelation)
        peak_lag = lags[peak_index]
        peak_value = self.autocorrelation[peak_index]

        self.confidence = peak_value / zero_lag_energy
        raw_bpm = 60.0 / (peak_lag * self.block_duration)

        if self.bpm == 0.0:
            self.bpm = raw_bpm
        else:
            self.bpm += self.smoothing * self.confidence * (raw_bpm - self.bpm)
