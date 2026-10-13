from collections import deque

import numpy as np

from arlx.audio.fft_analyzer import FrequencyBands, DEFAULT_FREQUENCY_BANDS

DEFAULT_WINDOW_DURATION = 0.5
DEFAULT_THRESHOLD_MULTIPLIER = 1.5
DEFAULT_REFRACTORY_DURATION = 0.05


class OnsetDetection:
    def __init__(self,
                 sample_rate: int = 44100,
                 block_size: int = 512,
                 frequency_bands: FrequencyBands = DEFAULT_FREQUENCY_BANDS,
                 window_duration: float = DEFAULT_WINDOW_DURATION,
                 threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER,
                 refractory_duration: float = DEFAULT_REFRACTORY_DURATION,
                 ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.frequency_bands = frequency_bands
        self.window_duration = window_duration
        self.threshold_multiplier = threshold_multiplier
        self.refractory_duration = refractory_duration

        self.block_duration = self.block_size / self.sample_rate
        self.history_length = max(int(self.window_duration / self.block_duration), 1)
        self.refractory_blocks = max(int(self.refractory_duration / self.block_duration), 1)

        num_bands = len(self.frequency_bands.bands)

        self.global_flux_history = deque(maxlen=self.history_length)
        self.band_flux_history = deque(maxlen=self.history_length)

        self.global_threshold = 0.0
        self.band_thresholds = np.zeros(num_bands)

        self.global_onset = False
        self.band_onsets = np.zeros(num_bands, dtype=bool)

        self._prev_global_above = False
        self._prev_band_above = np.zeros(num_bands, dtype=bool)

        self._global_refractory_countdown = 0
        self._band_refractory_countdown = np.zeros(num_bands, dtype=int)

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.block_duration = self.block_size / self.sample_rate
        self.history_length = max(int(self.window_duration / self.block_duration), 1)
        self.refractory_blocks = max(int(self.refractory_duration / self.block_duration), 1)
        self.global_flux_history = deque(maxlen=self.history_length)
        self.band_flux_history = deque(maxlen=self.history_length)

    def get_block_size(self):
        return self.block_size

    def set_block_size(self, block_size: int):
        self.block_size = block_size
        self.block_duration = self.block_size / self.sample_rate
        self.history_length = max(int(self.window_duration / self.block_duration), 1)
        self.refractory_blocks = max(int(self.refractory_duration / self.block_duration), 1)
        self.global_flux_history = deque(maxlen=self.history_length)
        self.band_flux_history = deque(maxlen=self.history_length)

    def get_frequency_bands(self) -> FrequencyBands:
        return self.frequency_bands

    # NOTE: also resets band_flux_history, not just the band-shaped arrays -
    # old entries in there are sized to the previous band count, and mixing
    # them with new-shaped entries breaks np.array(band_flux_history) once
    # warmed_up (ragged/inhomogeneous shape)
    def set_frequency_bands(self, frequency_bands: FrequencyBands):
        self.frequency_bands = frequency_bands
        num_bands = len(self.frequency_bands.bands)
        self.band_thresholds = np.zeros(num_bands)
        self.band_onsets = np.zeros(num_bands, dtype=bool)
        self._prev_band_above = np.zeros(num_bands, dtype=bool)
        self._band_refractory_countdown = np.zeros(num_bands, dtype=int)
        self.band_flux_history = deque(maxlen=self.history_length)

    def get_window_duration(self) -> float:
        return self.window_duration

    # NOTE: resizes the flux-history deques to the new window, keeping whatever
    # history already fits rather than wiping it (unlike the band-count resets
    # elsewhere, this doesn't change the shape of what's stored, just how much)
    def set_window_duration(self, window_duration: float):
        self.window_duration = window_duration
        self.history_length = max(int(self.window_duration / self.block_duration), 1)
        self.global_flux_history = deque(self.global_flux_history, maxlen=self.history_length)
        self.band_flux_history = deque(self.band_flux_history, maxlen=self.history_length)

    def get_threshold_multiplier(self) -> float:
        return self.threshold_multiplier

    def set_threshold_multiplier(self, threshold_multiplier: float):
        self.threshold_multiplier = threshold_multiplier

    def get_refractory_duration(self) -> float:
        return self.refractory_duration

    def set_refractory_duration(self, refractory_duration: float):
        self.refractory_duration = refractory_duration
        self.refractory_blocks = max(int(refractory_duration / self.block_duration), 1)

    def get_global_onset(self) -> bool:
        return self.global_onset

    def get_band_onsets(self) -> np.ndarray:
        return self.band_onsets.copy()

    def get_global_threshold(self) -> float:
        return self.global_threshold

    def get_band_thresholds(self) -> np.ndarray:
        return self.band_thresholds.copy()

    def process(self, global_flux: float, band_fluxes: np.ndarray) -> None:
        self.global_onset = self._process_global(global_flux)
        self.band_onsets = self._process_bands(band_fluxes)

    def _process_global(self, value: float) -> bool:
        warmed_up = len(self.global_flux_history) == self.history_length

        if warmed_up:
            history = np.array(self.global_flux_history)
            self.global_threshold = np.mean(history) + self.threshold_multiplier * np.std(history)
        above = warmed_up and value > self.global_threshold

        onset = False
        if self._global_refractory_countdown > 0:
            self._global_refractory_countdown -= 1
        elif above and not self._prev_global_above:
            onset = True
            self._global_refractory_countdown = self.refractory_blocks

        self._prev_global_above = above
        self.global_flux_history.append(value)
        return onset

    def _process_bands(self, values: np.ndarray) -> np.ndarray:
        warmed_up = len(self.band_flux_history) == self.history_length

        if warmed_up:
            history = np.array(self.band_flux_history)
            self.band_thresholds = np.mean(history, axis=0) + self.threshold_multiplier * np.std(history, axis=0)
        above = np.full(values.shape, warmed_up) & (values > self.band_thresholds)

        onsets = np.zeros_like(above)
        ready = self._band_refractory_countdown == 0
        triggered = ready & above & ~self._prev_band_above
        onsets[triggered] = True

        self._band_refractory_countdown[~ready] -= 1
        self._band_refractory_countdown[triggered] = self.refractory_blocks

        self._prev_band_above = above
        self.band_flux_history.append(values)
        return onsets
