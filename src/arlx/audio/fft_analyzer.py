
from dataclasses import dataclass

import numpy as np

@dataclass
class FrequencyBand:
    name: str
    low_freq: float
    high_freq: float

    def __post_init__(self):
        if self.low_freq >= self.high_freq:
            raise ValueError(f"Low frequency {self.low_freq} must be less than high frequency {self.high_freq} for band '{self.name}'") 

    def __lt__(self, other):
        if not isinstance(other, FrequencyBand):
            return NotImplemented
        return self.high_freq <= other.low_freq

    def __gt__(self, other):
        if not isinstance(other, FrequencyBand):
            return NotImplemented
        return self.low_freq >= other.high_freq

    def __eq__(self, other):
        if not isinstance(other, FrequencyBand):
            return NotImplemented
        return self.low_freq == other.low_freq and self.high_freq == other.high_freq

    def __repr__(self):
        return f"FrequencyBand(name='{self.name}', low_freq={self.low_freq}, high_freq={self.high_freq})"
    
@dataclass
class FrequencyBands:
    bands: list[FrequencyBand]

    def __post_init__(self):
        self.bands = sorted(self.bands, key=lambda band: band.low_freq)

        for current, following in zip(self.bands, self.bands[1:]):
            if not (current < following):
                raise ValueError(
                    f"Frequency bands overlap: '{current.name}' ({current.low_freq}-{current.high_freq}) "
                    f"and '{following.name}' ({following.low_freq}-{following.high_freq})"
                )

DEFAULT_FREQUENCY_BANDS = FrequencyBands([
    FrequencyBand("Sub-bass", 20, 60),
    FrequencyBand("Bass", 60, 250),
    FrequencyBand("Low Mid", 250, 500),
    FrequencyBand("Mid", 500, 2000),
    FrequencyBand("High Mid", 2000, 4000),
    FrequencyBand("Presence", 4000, 6000),
    FrequencyBand("Brilliance", 6000, 20000),
])

FLOOR_DB = -60
ATTACK_DB_PER_BLOCK = 3.0  
DECAY_DB_PER_BLOCK = 0.3

SMOOTH_ATTACK = 0.5  # fraction of the gap to close per block when the signal rises; tune by ear
SMOOTH_DECAY = 0.05  # fraction of the gap to close per block when the signal falls; tune by ear

class FFTAnalyzer:
    def __init__(self, 
                 sample_rate: int, 
                 block_size: int, 
                 frequency_bands: FrequencyBands = DEFAULT_FREQUENCY_BANDS,
                 smooth_attack: float = SMOOTH_ATTACK,
                 smooth_decay: float = SMOOTH_DECAY,
                 attack_db_per_block: float = ATTACK_DB_PER_BLOCK,
                 decay_db_per_block: float = DECAY_DB_PER_BLOCK,
                 floor_db: float = FLOOR_DB
                 ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.frequency_bands = frequency_bands
        self.smooth_attack = smooth_attack
        self.smooth_decay = smooth_decay
        self.attack_db_per_block = attack_db_per_block
        self.decay_db_per_block = decay_db_per_block
        self.floor_db = floor_db

        self.raw_fft_data = np.zeros(block_size // 2)

        self.mean_fft_bins = np.zeros(7)
        self.rms_fft_bins = np.zeros(7)
        self.log_rms_fft_bins = np.zeros(7)
        self.norm_fft_bins = np.zeros(7)
        self.smoothed_norm_bins = np.zeros(7)

        self.running_max_db = np.full(7, FLOOR_DB, dtype=float) 

        self.spec_x = np.fft.fftfreq(self.block_size, d=1.0 / self.sample_rate)[:self.block_size // 2]

    def get_raw_fft_data(self) -> np.ndarray:
        return self.raw_fft_data.copy()

    def get_mean_fft_bins(self) -> np.ndarray:
        return self.mean_fft_bins.copy()

    def get_rms_fft_bins(self) -> np.ndarray:
        return self.rms_fft_bins.copy()

    def get_log_rms_fft_bins(self) -> np.ndarray:
        return self.log_rms_fft_bins.copy()

    def get_norm_fft_bins(self) -> np.ndarray:
        return self.norm_fft_bins.copy()

    def get_smoothed_norm_bins(self) -> np.ndarray:
        return self.smoothed_norm_bins.copy()

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        fft_result = np.fft.fft(audio_block)
        self.raw_fft_data = np.abs(fft_result[:self.block_size // 2])

        for i, band in enumerate(self.frequency_bands.bands):
            band_indices = np.where((self.spec_x >= band.low_freq) & (self.spec_x < band.high_freq))[0]

            if len(band_indices) > 0:
                self.mean_fft_bins[i] = np.mean(self.raw_fft_data[band_indices])
                self.rms_fft_bins[i] = np.sqrt(np.mean(self.raw_fft_data[band_indices] ** 2))
                self.log_rms_fft_bins[i] = 20 * np.log10(max(self.rms_fft_bins[i], 1e-6))

                if self.log_rms_fft_bins[i] > self.running_max_db[i]:
                    self.running_max_db[i] = min(self.running_max_db[i] + self.attack_db_per_block, self.log_rms_fft_bins[i])
                else:
                    self.running_max_db[i] = max(self.running_max_db[i] - self.decay_db_per_block, self.floor_db)
                self.norm_fft_bins[i] = np.clip(
                    (self.log_rms_fft_bins[i] - self.floor_db) / (self.running_max_db[i] - self.floor_db),
                    0, 1
                ) if self.running_max_db[i] > self.floor_db else 0

                if self.norm_fft_bins[i] > self.smoothed_norm_bins[i]:
                    self.smoothed_norm_bins[i] += self.smooth_attack * (self.norm_fft_bins[i] - self.smoothed_norm_bins[i])
                else:
                    self.smoothed_norm_bins[i] += self.smooth_decay * (self.norm_fft_bins[i] - self.smoothed_norm_bins[i])

            else:
                self.mean_fft_bins[i] = 0
                self.rms_fft_bins[i] = 0
                self.log_rms_fft_bins[i] = self.floor_db
                self.running_max_db[i] = max(self.running_max_db[i] - self.decay_db_per_block, self.floor_db)
                self.norm_fft_bins[i] = 0
                self.smoothed_norm_bins[i] += self.smooth_decay * (0 - self.smoothed_norm_bins[i])