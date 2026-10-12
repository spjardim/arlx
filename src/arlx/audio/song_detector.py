

from typing import Deque

import numpy as np


class SongDetector:
    def __init__(self, dejavu_instance, record_seconds=5,sample_rate=44100, block_size=1024):
        self.dejavu = dejavu_instance
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.block_needed = int(record_seconds * sample_rate / block_size)
        self.blocks = Deque(maxlen=self.block_needed)
        
    def fingerprint_directory(self, directory_path, extensions):
        self.dejavu.fingerprint_directory(directory_path, extensions)

    def add_samples(self, samples):
        self.blocks.append(self._to_int16(samples))

    def detect_song_from_samples(self, samples):
        return self._generate_and_match(self._to_int16(samples))

    def detect_song_from_blocks(self):
        if len(self.blocks) < self.block_needed:
            return None  # Not enough data yet
        return self._generate_and_match(np.concatenate(list(self.blocks)))

    def _generate_and_match(self, samples_int16):
        hashes, _ = self.dejavu.generate_fingerprints(samples_int16, Fs=self.sample_rate)
        matches, dedup_hashes, _ = self.dejavu.find_matches(hashes)
        return self.dejavu.align_matches(matches, dedup_hashes, len(hashes))

    @staticmethod
    def _to_int16(samples):
        # Mic capture gives float32 in [-1, 1]; dejavu's fingerprint() expects
        # int16-scale PCM, same as what it reads from files.
        return np.clip(np.asarray(samples) * 32767, -32768, 32767).astype(np.int16)