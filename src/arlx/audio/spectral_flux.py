

import numpy as np

from arlx.audio.fft_analyzer import FrequencyBands, DEFAULT_FREQUENCY_BANDS

class SpectralFlux:
    def __init__(
        self,
        sample_rate: int,
        frame_size: int = 1024,
        hop_size: int = 512,
        frequency_bands: FrequencyBands = DEFAULT_FREQUENCY_BANDS,
        half_wave_rectify: bool = True,
        use_log_magnitude: bool = True,
        normalize_by_bin_count: bool = True,
        log_epsilon: float = 1e-6,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.frequency_bands = frequency_bands
        self.half_wave_rectify = half_wave_rectify
        self.use_log_magnitude = use_log_magnitude
        self.normalize_by_bin_count = normalize_by_bin_count
        self.log_epsilon = log_epsilon

        self.spec_x = np.fft.fftfreq(frame_size, d=1.0 / sample_rate)[:frame_size // 2]

        self.prev_spectrum = None
        self.global_flux = 0.0
        self.band_fluxes = np.zeros(len(frequency_bands.bands))

    def _to_diff_scale(self, spectrum: np.ndarray) -> np.ndarray:
        if self.use_log_magnitude:
            return 20 * np.log10(np.maximum(spectrum, self.log_epsilon))
        return spectrum

    def _reduce(self, squared_diff: np.ndarray) -> float:
        if len(squared_diff) == 0:
            return 0.0
        return np.mean(squared_diff) if self.normalize_by_bin_count else np.sum(squared_diff)

    def process(self, spectrum: np.ndarray) -> None:
        if self.prev_spectrum is None:
            self.prev_spectrum = spectrum

        current = self._to_diff_scale(spectrum)
        previous = self._to_diff_scale(self.prev_spectrum)

        diff = current - previous
        if self.half_wave_rectify:
            diff = np.maximum(diff, 0)

        self.global_flux = self._reduce(diff ** 2)

        band_fluxes = []
        for band in self.frequency_bands.bands:
            band_indices = np.where((self.spec_x >= band.low_freq) & (self.spec_x < band.high_freq))[0]
            band_fluxes.append(self._reduce(diff[band_indices] ** 2))
        self.band_fluxes = np.array(band_fluxes)

        self.prev_spectrum = spectrum

    def get_global_flux(self) -> float:
        return self.global_flux

    def get_band_fluxes(self) -> np.ndarray:
        return self.band_fluxes

    def update_frequency_bands(self, new_frequency_bands: FrequencyBands):
        self.frequency_bands = new_frequency_bands