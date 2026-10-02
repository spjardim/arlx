from dataclasses import dataclass
from typing import Optional

import numpy as np

from arlx.audio.audio_feature_tiers import RawAudioFeatures

# Rough perceptual band boundaries (Hz) used to bucket spectral energy for
# lighting effects (kick/bass -> low end, cymbals/hats -> high end).
BAND_EDGES_HZ: dict[str, tuple[float, float]] = {
    "bass": (20.0, 150.0),
    "low_mid": (150.0, 400.0),
    "mid": (400.0, 2600.0),
    "high_mid": (2600.0, 5200.0),
    "treble": (5200.0, 20000.0),
}

# Onset detector tuning.
ONSET_FLUX_EMA_ALPHA = 0.1  # how fast the "recent normal" flux baseline adapts
ONSET_SENSITIVITY = 1.5  # flux must exceed baseline * sensitivity to count as a hit
ONSET_MIN_FLUX = 1e-6  # ignore flux noise during near-silence
ONSET_REFRACTORY_SECONDS = 0.1  # minimum gap between two onsets

SPECTRAL_ROLLOFF_PERCENTAGE = 0.85  # fraction of total spectral energy below the rolloff frequency
FLATNESS_EPSILON = 1e-10  # avoids log(0)/div-by-0 when a block is pure silence



class RawAudioFeatureExtractor:
    """Extracts per-block lighting-relevant features from mono PCM samples.

    Stateful across calls: spectral flux and onset detection both compare
    the current block against the previous one, so a single instance must
    be fed consecutive blocks from one audio stream.
    """

    def __init__(self, sample_rate: float, block_size: int):
        self.sample_rate = sample_rate
        self.block_size = block_size

        self._window = np.hanning(block_size)
        self._window_norm = np.sum(self._window) / 2
        self._freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)

        self._previous_magnitude: Optional[np.ndarray] = None
        self._flux_baseline = 0.0
        self._blocks_since_onset = 0
        self._refractory_blocks = max(1, round(ONSET_REFRACTORY_SECONDS * sample_rate / block_size))

    def extract(self, samples: np.ndarray) -> RawAudioFeatures:
        mono = samples[:, 0] if samples.ndim > 1 else samples
        mono = mono.astype(np.float64, copy=False)

        rms = float(np.sqrt(np.mean(np.square(mono))))
        peak = float(np.max(np.abs(mono)))

        magnitude = np.abs(np.fft.rfft(mono * self._window)) / self._window_norm

        band_energy = {
            name: self._band_energy(magnitude, low_hz, high_hz)
            for name, (low_hz, high_hz) in BAND_EDGES_HZ.items()
        }
        spectral_centroid = self._spectral_centroid(magnitude)
        spectral_rolloff = self._spectral_rolloff(magnitude)
        spectral_flatness = self._spectral_flatness(magnitude)
        spectral_flux = self._spectral_flux(magnitude)
        onset = self._detect_onset(spectral_flux)

        self._previous_magnitude = magnitude

        return RawAudioFeatures(
            rms=rms,
            peak=peak,
            band_energy=RawAudioBands(**band_energy),
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_flatness=spectral_flatness,
            spectral_flux=spectral_flux,
            onset=onset,
        )

    def _band_energy(self, magnitude: np.ndarray, low_hz: float, high_hz: float) -> float:
        mask = (self._freqs >= low_hz) & (self._freqs < high_hz)
        if not np.any(mask):
            return 0.0
        return float(np.sqrt(np.mean(np.square(magnitude[mask]))))

    def _spectral_centroid(self, magnitude: np.ndarray) -> float:
        total_magnitude = np.sum(magnitude)
        if total_magnitude <= 0:
            return 0.0
        return float(np.sum(self._freqs * magnitude) / total_magnitude)

    def _spectral_rolloff(self, magnitude: np.ndarray) -> float:
        power = np.square(magnitude)
        total_power = np.sum(power)
        if total_power <= 0:
            return 0.0
        cumulative_power = np.cumsum(power)
        rolloff_bin = np.searchsorted(cumulative_power, SPECTRAL_ROLLOFF_PERCENTAGE * total_power)
        rolloff_bin = min(rolloff_bin, len(self._freqs) - 1)
        return float(self._freqs[rolloff_bin])

    def _spectral_flatness(self, magnitude: np.ndarray) -> float:
        power = np.square(magnitude) + FLATNESS_EPSILON
        geometric_mean = np.exp(np.mean(np.log(power)))
        arithmetic_mean = np.mean(power)
        return float(geometric_mean / arithmetic_mean)

    def _spectral_flux(self, magnitude: np.ndarray) -> float:
        if self._previous_magnitude is None:
            return 0.0
        difference = magnitude - self._previous_magnitude
        return float(np.sum(np.maximum(difference, 0.0)))

    def _detect_onset(self, spectral_flux: float) -> bool:
        self._blocks_since_onset += 1

        is_hit = (
            spectral_flux > ONSET_MIN_FLUX
            and spectral_flux > self._flux_baseline * ONSET_SENSITIVITY
            and self._blocks_since_onset >= self._refractory_blocks
        )

        self._flux_baseline = (
            ONSET_FLUX_EMA_ALPHA * spectral_flux + (1 - ONSET_FLUX_EMA_ALPHA) * self._flux_baseline
        )

        if is_hit:
            self._blocks_since_onset = 0

        return is_hit
