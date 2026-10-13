DEFAULT_CORRECTION_STRENGTH = 0.25
DEFAULT_BEAT_TOLERANCE = 0.2  # fraction of a beat period an onset may deviate from the grid and still correct it


class BeatPhaseTracker:
    def __init__(self,
                 sample_rate: int = 44100,
                 block_size: int = 512,
                 correction_strength: float = DEFAULT_CORRECTION_STRENGTH,
                 beat_tolerance: float = DEFAULT_BEAT_TOLERANCE,
                 ):
        self.sample_rate = sample_rate
        self.hop_length = block_size
        self.correction_strength = correction_strength
        self.beat_tolerance = beat_tolerance

        self.block_duration = block_size / sample_rate

        self.elapsed_time = 0.0
        self.period = 0.0
        self.next_beat_time = None

        self.phase = 0.0
        self.beat_now = False
        self.next_beat_in = 0.0

    def get_phase(self) -> float:
        return self.phase

    def get_beat_now(self) -> bool:
        return self.beat_now

    def get_next_beat_in(self) -> float:
        return self.next_beat_in

    def process(self, onset: bool, bpm: float) -> None:
        self.elapsed_time += self.block_duration
        self.beat_now = False

        if bpm <= 0:
            self.period = 0.0
            self.next_beat_time = None
            self.phase = 0.0
            self.next_beat_in = 0.0
            return

        self.period = 60.0 / bpm

        if self.next_beat_time is None:
            self.next_beat_time = self.elapsed_time + self.period

        if onset:
            deviation = ((self.elapsed_time - self.next_beat_time + self.period / 2) % self.period) - self.period / 2
            if abs(deviation) <= self.beat_tolerance * self.period:
                self.next_beat_time += self.correction_strength * deviation

        while self.elapsed_time >= self.next_beat_time:
            self.beat_now = True
            self.next_beat_time += self.period

        self.next_beat_in = max(self.next_beat_time - self.elapsed_time, 0.0)
        self.phase = 1.0 - min(self.next_beat_in / self.period, 1.0)
