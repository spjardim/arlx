from collections import deque

import numpy as np

DEFAULT_BEATS_PER_BAR = 4
DEFAULT_HISTORY_BARS = 8


class DownbeatDetector:
    def __init__(self,
                 beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
                 history_bars: int = DEFAULT_HISTORY_BARS,
                 ):
        self.beats_per_bar = beats_per_bar
        self.history_bars = history_bars

        self.position_strength_history = [deque(maxlen=self.history_bars) for _ in range(self.beats_per_bar)]

        self.beat_index = 0
        self.downbeat_offset = 0
        self.is_downbeat = False

    def get_beats_per_bar(self) -> int:
        return self.beats_per_bar

    # NOTE: changing the bucket count invalidates the beat_index -> bucket
    # mapping, so this resets position history/tracking state the same way a
    # band-count change resets FFTAnalyzer's bins
    def set_beats_per_bar(self, beats_per_bar: int):
        self.beats_per_bar = beats_per_bar
        self.position_strength_history = [deque(maxlen=self.history_bars) for _ in range(beats_per_bar)]
        self.beat_index = 0
        self.downbeat_offset = 0
        self.is_downbeat = False

    def get_history_bars(self) -> int:
        return self.history_bars

    def set_history_bars(self, history_bars: int):
        self.history_bars = history_bars
        self.position_strength_history = [
            deque(history, maxlen=history_bars) for history in self.position_strength_history
        ]

    def get_beat_index(self) -> int:
        return self.beat_index

    def get_downbeat_offset(self) -> int:
        return self.downbeat_offset

    def get_is_downbeat(self) -> bool:
        return self.is_downbeat

    def get_position_strengths(self) -> np.ndarray:
        return np.array([np.mean(h) if len(h) > 0 else 0.0 for h in self.position_strength_history])

    def process(self, beat_now: bool, beat_strength: float) -> None:
        self.is_downbeat = False
        if not beat_now:
            return

        self.position_strength_history[self.beat_index].append(beat_strength)

        self.downbeat_offset = int(np.argmax(self.get_position_strengths()))
        self.is_downbeat = (self.beat_index == self.downbeat_offset)

        self.beat_index = (self.beat_index + 1) % self.beats_per_bar
