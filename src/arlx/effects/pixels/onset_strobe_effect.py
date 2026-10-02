import numpy as np

from arlx.audio.audio_feature_tiers import AudioFeatures
from arlx.effects.pixels.base import PixelEffect


class OnsetStrobeEffect(PixelEffect):
    """Flashes the whole strip white on each detected onset, then fades
    out over flash_decay seconds. Flash rate is capped by max_strobe_hz
    (enforced by the base class) so a burst of onsets can't exceed a safe
    strobe rate."""

    def __init__(self, num_pixels: int, flash_decay: float = 0.15, **kwargs):
        super().__init__(num_pixels, **kwargs)
        self.flash_decay = flash_decay
        self._brightness = 0.0

    def render(self, features: AudioFeatures, dt: float) -> np.ndarray:
        flash_ready = self._flash_ready(dt)

        if features.tier0.onset and flash_ready:
            self._record_flash()
            self._brightness = 1.0
        else:
            self._brightness = max(0.0, self._brightness - dt / self.flash_decay)

        return np.tile([self._brightness] * 3, (self.num_pixels, 1))
