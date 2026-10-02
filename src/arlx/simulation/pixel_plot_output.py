import matplotlib.pyplot as plt
import numpy as np

from arlx.outputs.pixel.base import PixelOutput


class PixelPlotOutput(PixelOutput):
    """Live matplotlib preview of a pixel strip - a stand-in for real LED
    hardware so effects can be developed and watched without a strip
    wired up.
    """

    def __init__(self, num_pixels: int):
        self.num_pixels = num_pixels
        self._figure = None
        self._image = None

    def start(self) -> bool:
        plt.ion()
        self._figure, axis = plt.subplots(figsize=(10, 1))
        axis.set_axis_off()
        self._image = axis.imshow(np.zeros((1, self.num_pixels, 3)), aspect="auto")
        self._figure.show()
        return True

    def stop(self) -> bool:
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
            self._image = None
        return True

    def send(self, pixels: np.ndarray) -> None:
        if self._image is None:
            return
        self._image.set_data(np.clip(pixels, 0.0, 1.0).reshape(1, self.num_pixels, 3))
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
