import colorsys

import numpy as np

from arlx.audio.audio_feature_tiers import AudioFeatures, RawAudioBands
from arlx.effects.pixels.band_hues import BAND_HUES
from arlx.effects.pixels.base import PixelEffect


class BandColorEffect(PixelEffect):
    """Whole-strip color wash: brightness tracks RMS, hue tracks which
    frequency band currently carries the most energy."""

    def __init__(self, num_pixels: int, gain: float = 6.0, **kwargs):
        super().__init__(num_pixels, **kwargs)
        self.gain = gain

    def render(self, features: AudioFeatures, dt: float) -> np.ndarray:
        raw = features.tier0
        brightness = min(1.0, raw.rms * self.gain)
        hue = self._band_weighted_hue(raw.band_energy)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, brightness)
        return np.tile(rgb, (self.num_pixels, 1))

    def _band_weighted_hue(self, band_energy: RawAudioBands) -> float:
        total_energy = (
            band_energy.bass
            + band_energy.low_mid
            + band_energy.mid
            + band_energy.high_mid
            + band_energy.treble
        )
        if total_energy <= 0:
            return BAND_HUES["bass"]

        return (
            band_energy.bass * BAND_HUES["bass"]
            + band_energy.low_mid * BAND_HUES["low_mid"]
            + band_energy.mid * BAND_HUES["mid"]
            + band_energy.high_mid * BAND_HUES["high_mid"]
            + band_energy.treble * BAND_HUES["treble"]
        ) / total_energy
