import colorsys

import numpy as np

from arlx.audio.audio_feature_tiers import AudioFeatures
from arlx.effects.pixels.band_hues import BAND_HUES
from arlx.effects.pixels.base import PixelEffect

BAND_ORDER = ["bass", "low_mid", "mid", "high_mid", "treble"]


class BandMeterEffect(PixelEffect):
    """Splits the strip into one contiguous segment per frequency band - a
    spatial VU meter, as opposed to BandColorEffect blending all bands into
    one color across the whole strip."""

    def __init__(self, num_pixels: int, gain: float = 6.0, **kwargs):
        super().__init__(num_pixels, **kwargs)
        self.gain = gain
        self._segment_bounds = self._compute_segment_bounds()

    def _compute_segment_bounds(self) -> list[tuple[int, int]]:
        edges = np.linspace(0, self.num_pixels, len(BAND_ORDER) + 1).astype(int)
        return list(zip(edges[:-1], edges[1:]))

    def render(self, features: AudioFeatures, dt: float) -> np.ndarray:
        band_energy = features.tier0.band_energy
        pixels = np.zeros((self.num_pixels, 3))

        for band_name, (start, end) in zip(BAND_ORDER, self._segment_bounds):
            brightness = min(1.0, getattr(band_energy, band_name) * self.gain)
            pixels[start:end] = colorsys.hsv_to_rgb(BAND_HUES[band_name], 1.0, brightness)

        return pixels
