from collections import deque
from pathlib import Path

import numpy as np

import dejavu.logic.decoder as decoder

class SongDetector:
    def __init__(self, 
                 dejavu_instance, 
                 record_seconds=5, 
                 sample_rate=44100, 
                 block_size=1024,
                 detect_every_seconds=1.0, 
                 sync_check_seconds=1.0, 
                 sync_check_every_seconds=0.5,
                 sync_confidence_threshold=0.5):
        self.dejavu = dejavu_instance
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.detect_every_seconds = detect_every_seconds
        self.block_duration = block_size / sample_rate
        self.record_seconds = record_seconds
        self.sync_check_seconds = sync_check_seconds
        self.sync_check_every_seconds = sync_check_every_seconds
        self.sync_confidence_threshold = sync_confidence_threshold

        self.block_needed = int(self.record_seconds * self.sample_rate / self.block_size)
        self.detect_every_blocks = max(int(self.detect_every_seconds / self.block_duration), 1)
        self.blocks = deque(maxlen=self.block_needed)
        self._block_count = 0

        # Once a song is known, we stop calling dejavu every cycle and switch
        # to a much cheaper check: does recent live audio still correlate with
        # the known reference at our estimated position? No spectrogram,
        # hashing, or DB lookup - just a direct waveform comparison.
        self.sync_check_blocks_needed = max(int(self.sync_check_seconds * self.sample_rate / self.block_size), 1)
        self.sync_check_every_blocks = max(int(self.sync_check_every_seconds / self.block_duration), 1)
        self.sync_check_buffer = deque(maxlen=self.sync_check_blocks_needed)
        self.sync_confidence = 0.0

        self.songs_directory = None
        self.extensions = None

        # Position tracking: time_since_anchor resets to 0 on every successful
        # detection or sync correction, so current position is always "where
        # we last confirmed, plus however long it's been".
        self.position_at_anchor = None
        self.time_since_anchor = 0.0
        self.current_position_estimate = None

        self.current_song_id = None
        self.current_song_name = None
        self.current_song_samples = None
        self.current_song_fs = None

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._recompute_derived()

    def get_block_size(self):
        return self.block_size

    def set_block_size(self, block_size: int):
        self.block_size = block_size
        self._recompute_derived()

    # NOTE: every block-count derived from sample_rate/block_size in __init__
    # (block_needed, detect_every_blocks, sync_check_blocks_needed,
    # sync_check_every_blocks) plus the two deques sized off them
    def _recompute_derived(self):
        self.block_duration = self.block_size / self.sample_rate
        self.block_needed = int(self.record_seconds * self.sample_rate / self.block_size)
        self.detect_every_blocks = max(int(self.detect_every_seconds / self.block_duration), 1)
        self.sync_check_blocks_needed = max(int(self.sync_check_seconds * self.sample_rate / self.block_size), 1)
        self.sync_check_every_blocks = max(int(self.sync_check_every_seconds / self.block_duration), 1)
        self.blocks = deque(self.blocks, maxlen=self.block_needed)
        self.sync_check_buffer = deque(self.sync_check_buffer, maxlen=self.sync_check_blocks_needed)

    def get_record_seconds(self) -> float:
        return self.record_seconds

    def set_record_seconds(self, record_seconds: float):
        self.record_seconds = record_seconds
        self.block_needed = int(record_seconds * self.sample_rate / self.block_size)
        self.blocks = deque(self.blocks, maxlen=self.block_needed)

    def get_detect_every_seconds(self) -> float:
        return self.detect_every_seconds

    def set_detect_every_seconds(self, detect_every_seconds: float):
        self.detect_every_seconds = detect_every_seconds
        self.detect_every_blocks = max(int(detect_every_seconds / self.block_duration), 1)

    def get_sync_check_seconds(self) -> float:
        return self.sync_check_seconds

    def set_sync_check_seconds(self, sync_check_seconds: float):
        self.sync_check_seconds = sync_check_seconds
        self.sync_check_blocks_needed = max(int(sync_check_seconds * self.sample_rate / self.block_size), 1)
        self.sync_check_buffer = deque(self.sync_check_buffer, maxlen=self.sync_check_blocks_needed)

    def get_sync_check_every_seconds(self) -> float:
        return self.sync_check_every_seconds

    def set_sync_check_every_seconds(self, sync_check_every_seconds: float):
        self.sync_check_every_seconds = sync_check_every_seconds
        self.sync_check_every_blocks = max(int(sync_check_every_seconds / self.block_duration), 1)

    def get_sync_confidence_threshold(self) -> float:
        return self.sync_confidence_threshold

    def set_sync_confidence_threshold(self, sync_confidence_threshold: float):
        self.sync_confidence_threshold = sync_confidence_threshold

    def fingerprint_directory(self, directory_path, extensions):
        self.dejavu.fingerprint_directory(directory_path, extensions)
        self.songs_directory = Path(directory_path)
        self.extensions = extensions

    def process(self, sample):
        self.time_since_anchor += self.block_duration
        self._block_count += 1

        if self.position_at_anchor is not None:
            self.current_position_estimate = self.position_at_anchor + self.time_since_anchor

        self.blocks.append(self._to_int16(sample))
        self.sync_check_buffer.append(np.asarray(sample, dtype=np.float32))

        if self.current_song_id is None:
            self._try_detect()
        else:
            self._try_sync_check()

    def _try_detect(self):
        if len(self.blocks) < self.block_needed:
            return
        if self._block_count % self.detect_every_blocks != 0:
            return

        results = self._generate_and_match(np.concatenate(list(self.blocks)))
        if not results:
            return

        top = results[0]
        song_id = top["song_id"]
        song_name = top["song_name"].decode()

        # Load the reference file (if it's a new song) BEFORE publishing the
        # position estimate - otherwise a reader could see a fresh position
        # estimate with no reference data loaded yet.
        if song_id != self.current_song_id:
            self._load_reference(song_name)
            self.current_song_id = song_id
            self.current_song_name = song_name

        # offset_seconds is where the START of the fingerprinted window (the
        # last record_seconds of audio) sits in the reference track - "now"
        # is the END of that window.
        self.position_at_anchor = top["offset_seconds"] + self.record_seconds
        self.time_since_anchor = 0.0
        self.current_position_estimate = self.position_at_anchor
        self.sync_confidence = 1.0  # just confirmed by a real dejavu match

    def _try_sync_check(self):
        if len(self.sync_check_buffer) < self.sync_check_blocks_needed:
            return
        if self._block_count % self.sync_check_every_blocks != 0:
            return

        offset_seconds, confidence = self._best_sync_offset()
        self.sync_confidence = confidence

        if confidence >= self.sync_confidence_threshold:
            # Cheap correction using the best-matching lag, same idea as
            # BeatPhaseTracker nudging toward real onsets - no dejavu needed.
            self.position_at_anchor = self.current_position_estimate + offset_seconds
            self.time_since_anchor = 0.0
            self.current_position_estimate = self.position_at_anchor
        else:
            # Lost sync (song ended, or a DJ is mixing in something else) -
            # drop back to full re-detection rather than trust a stale guess.
            self.current_song_id = None
            self.current_song_name = None
            self.current_song_samples = None
            self.current_song_fs = None
            self.position_at_anchor = None
            self.current_position_estimate = None

    def _best_sync_offset(self, max_lag_seconds=0.3, lag_step_seconds=0.01):
        # Raw waveform correlation has an extremely narrow peak - even a few
        # milliseconds of misalignment collapses it to near zero (verified
        # empirically), which a coarse/cheap lag search would almost always
        # miss. Comparing amplitude envelopes instead degrades gradually with
        # misalignment, so a coarse search actually finds it.
        live = np.concatenate(list(self.sync_check_buffer))
        live_envelope = self._envelope(live, lag_step_seconds)
        live_centered = live_envelope - np.mean(live_envelope)
        live_energy = np.sqrt(np.sum(live_centered ** 2))
        if live_energy == 0:
            return 0.0, 0.0

        fs = self.current_song_fs
        center_sample = int(self.current_position_estimate * fs) - len(live)
        max_lag_samples = int(max_lag_seconds * fs)
        lag_step_samples = max(int(lag_step_seconds * fs), 1)

        best_confidence = 0.0
        best_offset_seconds = 0.0
        for lag in range(-max_lag_samples, max_lag_samples + 1, lag_step_samples):
            start = center_sample + lag
            end = start + len(live)
            if start < 0 or end > len(self.current_song_samples):
                continue

            reference = self.current_song_samples[start:end]
            ref_envelope = self._envelope(reference, lag_step_seconds)
            ref_centered = ref_envelope - np.mean(ref_envelope)
            ref_energy = np.sqrt(np.sum(ref_centered ** 2))
            if ref_energy == 0:
                continue

            correlation = np.dot(live_centered, ref_centered) / (live_energy * ref_energy)
            if correlation > best_confidence:
                best_confidence = correlation
                best_offset_seconds = lag / fs

        return best_offset_seconds, max(best_confidence, 0.0)

    def _envelope(self, signal, resolution_seconds):
        window = max(int(resolution_seconds * self.sample_rate), 1)
        n_windows = len(signal) // window
        if n_windows == 0:
            return np.abs(signal)
        trimmed = signal[:n_windows * window].reshape(n_windows, window)
        return np.max(np.abs(trimmed), axis=1)

    def detect_song_from_samples(self, samples):
        """One-shot recognition from a standalone sample array (no rolling buffer/position tracking)."""
        return self._generate_and_match(self._to_int16(samples))

    def get_song_id(self):
        return self.current_song_id

    def get_song_name(self):
        return self.current_song_name

    def get_current_position_seconds(self):
        return self.current_position_estimate

    def get_sync_confidence(self):
        return self.sync_confidence

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
