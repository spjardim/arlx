import colorsys

import numpy as np

from arlx.audio.audio_feature_tiers import AudioFeatures
from arlx.effects.pixels.base import PixelEffect


class ChaseEffect(PixelEffect):
    """A single comet travels down the strip with a short fading tail.
    Speed is driven by treble energy on top of a constant base speed, so
    it's never fully still but visibly speeds up on hi-hats/cymbals."""

    def __init__(
        self,
        num_pixels: int,
        base_speed: float = 5.0,
        speed_gain: float = 40.0,
        tail_length: int = 6,
        hue: float = 0.55,
        **kwargs,
    ):
        super().__init__(num_pixels, **kwargs)
        self.base_speed = base_speed  # pixels/sec with no treble energy
        self.speed_gain = speed_gain  # extra pixels/sec per unit of treble energy
        self.tail_length = min(tail_length, num_pixels)
        self.hue = hue
        self._position = 0.0

    def render(self, features: AudioFeatures, dt: float) -> np.ndarray:
        speed = self.base_speed + features.tier0.band_energy.treble * self.speed_gain
        self._position = (self._position + speed * dt) % self.num_pixels

        pixels = np.zeros((self.num_pixels, 3))
        head = int(self._position)
        for offset in range(self.tail_length):
            brightness = 1.0 - (offset / self.tail_length)
            pixels[(head - offset) % self.num_pixels] = colorsys.hsv_to_rgb(self.hue, 1.0, brightness)

        return pixels
