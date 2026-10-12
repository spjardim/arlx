from collections import deque
from pathlib import Path

import numpy as np

import dejavu.logic.decoder as decoder


class SongDetector:
    def __init__(self, dejavu_instance, record_seconds=5, sample_rate=44100, block_size=1024,
                 detect_every_seconds=1.0):
        self.dejavu = dejavu_instance
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.block_duration = block_size / sample_rate
        self.record_seconds = record_seconds

        self.block_needed = int(record_seconds * sample_rate / block_size)
        self.detect_every_blocks = max(int(detect_every_seconds / self.block_duration), 1)
        self.blocks = deque(maxlen=self.block_needed)
        self._block_count = 0

        self.songs_directory = None
        self.extensions = None

        # Position tracking: time_since_anchor resets to 0 on every successful
        # detection (same song confirmed again, or a different song entirely -
        # the mechanism doesn't need to tell those apart), so current position
        # is always "where we last confirmed, plus however long it's been".
        self.position_at_anchor = None
        self.time_since_anchor = 0.0
        self.current_position_estimate = None

        self.current_song_name = None
        self.current_song_samples = None
        self.current_song_fs = None

    def fingerprint_directory(self, directory_path, extensions):
        self.dejavu.fingerprint_directory(directory_path, extensions)
        self.songs_directory = Path(directory_path)
        self.extensions = extensions

    def process(self, sample):
        self.blocks.append(self._to_int16(sample))
        self.time_since_anchor += self.block_duration
        self._block_count += 1

        if self.position_at_anchor is not None:
            self.current_position_estimate = self.position_at_anchor + self.time_since_anchor

        if len(self.blocks) < self.block_needed:
            return
        if self._block_count % self.detect_every_blocks != 0:
            return

        results = self._generate_and_match(np.concatenate(list(self.blocks)))
        if not results:
            return

        top = results[0]
        song_name = top["song_name"].decode()

        # Load the reference file (if it's a new song) BEFORE publishing the
        # position estimate - otherwise a reader could see a fresh position
        # estimate with no reference data loaded yet.
        if song_name != self.current_song_name:
            self._load_reference(song_name)
            self.current_song_name = song_name

        # offset_seconds is where the START of the fingerprinted window (the
        # last record_seconds of audio) sits in the reference track - "now"
        # is the END of that window.
        self.position_at_anchor = top["offset_seconds"] + self.record_seconds
        self.time_since_anchor = 0.0
        self.current_position_estimate = self.position_at_anchor

    def detect_song_from_samples(self, samples):
        """One-shot recognition from a standalone sample array (no rolling buffer/position tracking)."""
        return self._generate_and_match(self._to_int16(samples))

    def get_song_name(self):
        return self.current_song_name

    def get_current_position_seconds(self):
        return self.current_position_estimate

    def get_reference_window(self, window_seconds=1.0):
        """Slice of the known reference file spanning +/- window_seconds around
        the current position estimate. (None, None) if no song is identified yet."""
        if self.current_position_estimate is None or self.current_song_samples is None:
            return None, None

        position = self.current_position_estimate
        start_sample = int((position - window_seconds) * self.current_song_fs)
        end_sample = int((position + window_seconds) * self.current_song_fs)

        start_clamped = max(start_sample, 0)
        end_clamped = min(end_sample, len(self.current_song_samples))
        if start_clamped >= end_clamped:
            return None, None

        window = self.current_song_samples[start_clamped:end_clamped]
        t = np.linspace(
            (start_clamped - start_sample) / self.current_song_fs - window_seconds,
            window_seconds - (end_sample - end_clamped) / self.current_song_fs,
            len(window),
        )
        return t, window

    def _load_reference(self, song_name):
        if self.songs_directory is None:
            raise RuntimeError("fingerprint_directory() must be called before songs can be identified")

        for extension in self.extensions:
            ext = extension if extension.startswith(".") else f".{extension}"
            candidate = self.songs_directory / f"{song_name}{ext}"
            if candidate.exists():
                channels, fs, _ = decoder.read(str(candidate))
                self.current_song_samples = channels[0].astype(np.float32) / 32767.0
                self.current_song_fs = fs
                return

        raise FileNotFoundError(f"Could not find reference audio for '{song_name}' in {self.songs_directory}")

    def _generate_and_match(self, samples_int16):
        hashes, _ = self.dejavu.generate_fingerprints(samples_int16, Fs=self.sample_rate)
        matches, dedup_hashes, _ = self.dejavu.find_matches(hashes)
        return self.dejavu.align_matches(matches, dedup_hashes, len(hashes))

    @staticmethod
    def _to_int16(samples):
        # Mic capture gives float32 in [-1, 1]; dejavu's fingerprint() expects
        # int16-scale PCM, same as what it reads from files.
        return np.clip(np.asarray(samples) * 32767, -32768, 32767).astype(np.int16)
