from abc import ABC, abstractmethod

import numpy as np

from arlx.audio.audio_feature_tiers import AudioFeatures


class PixelEffect(ABC):
    """
    Base class for pixel (LED strip) effects.

    A PixelEffect is a pure render step: given the current audio features
    and the time elapsed since the last frame, it produces a color for
    every pixel on the strip it's bound to. It has no knowledge of where
    those colors are sent (E1.31, a plotted preview, ...) and no
    knowledge of whether it is currently the active effect - that's a
    Layer's job.

    render() returns an (num_pixels, 3) float array with channels in
    [0, 1]; scaling to whatever the output protocol needs (e.g. 0-255)
    happens downstream, not here.
    """

    def __init__(self, num_pixels: int, max_strobe_hz: float = 20.0):
        self.num_pixels = num_pixels
        self.max_strobe_hz = max_strobe_hz
        self._time_since_last_flash = float("inf")

    @abstractmethod
    def render(self, features: AudioFeatures, dt: float) -> np.ndarray:
        raise NotImplementedError

    def _flash_ready(self, dt: float) -> bool:
        """Rate-limits strobe-style behavior to max_strobe_hz.

        Call once per frame regardless of whether a flash actually
        happens - the clock has to advance every frame to measure real
        elapsed time. Returns True once enough time has passed since the
        last _record_flash() call. Checking readiness does not consume
        it; only _record_flash() does, so a frame that's ready but
        doesn't flash (e.g. no onset this frame) doesn't reset the clock.

        Lives here, not in individual effects, so every strobe-style
        effect enforces the same cap through one shared code path instead
        of each reimplementing (and potentially getting wrong) its own
        rate limiting.
        """
        self._time_since_last_flash += dt
        return self._time_since_last_flash >= (1.0 / self.max_strobe_hz)

    def _record_flash(self) -> None:
        """Call when a flash actually fires, to reset the rate limiter."""
        self._time_since_last_flash = 0.0
