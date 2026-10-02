from abc import ABC, abstractmethod

import numpy as np


class PixelOutput(ABC):
    """Base class for anything that can display a pixel strip's colors -
    real LED hardware (E1.31) or a simulated preview. Mirrors AudioSource's
    start/stop lifecycle on the other end of the pipeline.
    """

    @abstractmethod
    def start(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def send(self, pixels: np.ndarray) -> None:
        """pixels is an (num_pixels, 3) float array with channels in [0, 1]."""
        raise NotImplementedError
