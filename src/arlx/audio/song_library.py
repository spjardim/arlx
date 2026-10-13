from pathlib import Path

import librosa
import numpy as np
import psycopg2
import psycopg2.extras

import dejavu.logic.decoder as decoder

from arlx.audio.fft_analyzer import FFTAnalyzer, DEFAULT_FREQUENCY_BANDS
from arlx.audio.spectral_flux import SpectralFlux
from arlx.audio.onest_detection import OnsetDetection
from arlx.audio.downbeat_detector import DownbeatDetector

BLOCK_SIZE = 1024


class SongLibrary:
    """Owns 'adding a song': fingerprints it with dejavu, then precomputes and
    caches the full tier0 feature timeline for it, so nothing has to be
    computed live once SongDetector confirms which song is playing."""

    def __init__(self, dejavu_instance, db_config):
        self.dejavu = dejavu_instance
        self.connection = psycopg2.connect(**db_config)
        self.connection.autocommit = True
        self._create_table()

    def add_song(self, file_path):
        file_path = Path(file_path)
        self.dejavu.fingerprint_file(str(file_path))
        song_id = self._find_song_id(file_path.stem)

        features = self._precompute_features(file_path)
        self._store_features(song_id, features)
        return song_id

    def get_features(self, song_id):
        with self.connection.cursor() as cur:
            cur.execute("SELECT features FROM song_features WHERE song_id = %s", (song_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def _create_table(self):
        with self.connection.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS song_features (
                    song_id INTEGER PRIMARY KEY REFERENCES songs(song_id) ON DELETE CASCADE,
                    features JSONB NOT NULL,
                    computed_at TIMESTAMP NOT NULL DEFAULT now()
                );
            """)

    def _find_song_id(self, song_name):
        for song in self.dejavu.get_fingerprinted_songs():
            if song["song_name"] == song_name:
                return song["song_id"]
        raise RuntimeError(f"Could not find song_id for '{song_name}' after fingerprinting")

    def _precompute_features(self, file_path):
        channels, fs, _ = decoder.read(str(file_path))
        # Downmix to mono by averaging all channels, not just taking the
        # first one - we have the full stereo file here (unlike live mono mic
        # capture), so there's no reason to silently discard the other channel.
        samples = np.mean([channel.astype(np.float32) for channel in channels], axis=0) / 32767.0
        block_duration = BLOCK_SIZE / fs
        num_blocks = len(samples) // BLOCK_SIZE

        fft_analyzer = FFTAnalyzer(sample_rate=fs, block_size=BLOCK_SIZE)
        spectral_flux = SpectralFlux(sample_rate=fs, block_size=BLOCK_SIZE, hop_size=BLOCK_SIZE)
        onset_detection = OnsetDetection(sample_rate=fs, block_size=BLOCK_SIZE)

        band_names = [band.name for band in DEFAULT_FREQUENCY_BANDS.bands]
        raw_band_energy = np.zeros((num_blocks, 7))
        global_flux = np.zeros(num_blocks)
        overall_rms = np.zeros(num_blocks)
        overall_peak = np.zeros(num_blocks)
        onset_times = []
        band_onset_times = {name: [] for name in band_names}

        for i in range(num_blocks):
            block = samples[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            fft_analyzer.process(block)
            spectral_flux.process(fft_analyzer.get_raw_fft_data())
            onset_detection.process(spectral_flux.get_global_flux(), spectral_flux.get_band_fluxes())

            raw_band_energy[i] = fft_analyzer.get_mean_fft_bins()
            global_flux[i] = spectral_flux.get_global_flux()
            overall_rms[i] = np.sqrt(np.mean(block ** 2))
            overall_peak[i] = np.max(np.abs(block))

            band_onsets = onset_detection.get_band_onsets()
            if band_onsets[0] or band_onsets[1]:
                onset_times.append(round(i * block_duration, 4))
            for band_index, band_name in enumerate(band_names):
                if band_onsets[band_index]:
                    band_onset_times[band_name].append(round(i * block_duration, 4))

        band_energy = self._normalize_by_percentile(raw_band_energy)
        overall_rms_norm = self._normalize_by_percentile(overall_rms)
        overall_peak_norm = self._normalize_by_percentile(overall_peak)

        # Not TempoTracker/BeatPhaseTracker here - we verified those are
        # unreliable over a whole track (confidence check, test_file_17/18).
        # librosa's dynamic-programming beat tracker is the trustworthy source
        # for the offline/known-song case.
        tempo, beat_times_array = librosa.beat.beat_track(y=samples, sr=fs, units="time")
        tempo = float(np.asarray(tempo).reshape(-1)[0])
        beat_times = [round(float(t), 4) for t in beat_times_array]

        downbeat_times = self._find_downbeat_times(beat_times, global_flux, num_blocks, block_duration)
        structure = self._find_structure(raw_band_energy, block_duration)

        return {
            "duration_seconds": round(num_blocks * block_duration, 3),
            "bpm": round(tempo, 2),
            "beat_times": beat_times,
            "downbeat_times": downbeat_times,
            "onset_times": onset_times,
            "band_onset_times": band_onset_times,
            "overall_loudness": {
                "resolution_seconds": round(block_duration, 6),
                "rms": overall_rms_norm.round(4).tolist(),
                "peak": overall_peak_norm.round(4).tolist(),
            },
            "band_energy": {
                "resolution_seconds": round(block_duration, 6),
                "band_names": band_names,
                "bands": band_energy.round(4).tolist(),
            },
            "structure": structure,
        }

    @staticmethod
    def _normalize_by_percentile(raw_values, percentile=95):
        # Live FFTAnalyzer normalizes against a causal running max, which has
        # to "catch up" over time. Offline, we already know the track's true
        # loudness ceiling (per band, or overall), so normalize against that
        # directly - a percentile rather than the true max, so one transient
        # spike doesn't squash the normalization for the rest of the track.
        ceiling = np.percentile(raw_values, percentile, axis=0)
        ceiling = np.where(ceiling == 0, 1.0, ceiling)  # avoid divide-by-zero for silence
        return np.clip(raw_values / ceiling, 0, 1)

    @staticmethod
    def _find_downbeat_times(beat_times, global_flux, num_blocks, block_duration):
        # DownbeatDetector itself is unchanged - it just needed a trustworthy
        # beat grid, which librosa's beat_times now provides instead of our
        # own (unreliable, for a full track) BeatPhaseTracker.
        beat_block_flags = np.zeros(num_blocks, dtype=bool)
        for t in beat_times:
            block_index = int(t / block_duration)
            if 0 <= block_index < num_blocks:
                beat_block_flags[block_index] = True

        downbeat_detector = DownbeatDetector()
        downbeat_times = []
        for i in range(num_blocks):
            downbeat_detector.process(bool(beat_block_flags[i]), float(global_flux[i]))
            if beat_block_flags[i] and downbeat_detector.get_is_downbeat():
                downbeat_times.append(round(i * block_duration, 4))
        return downbeat_times

    @staticmethod
    def _find_structure(raw_band_energy, block_duration, frame_seconds=1.0,
                         kernel_radius_frames=8, min_gap_seconds=4.0, drop_ratio=1.3, build_frames=4):
        """Structural boundary detection via self-similarity novelty (Foote's
        method): coarsen band energy to ~1 frame/second, build a cosine
        self-similarity matrix between every pair of frames, then convolve a
        checkerboard kernel along its diagonal - high response exactly where
        the audio just before a point is internally consistent, the audio
        just after is internally consistent, but the two differ from each
        other. Peaks in that "novelty curve" are candidate structural
        boundaries; each is then classified as a drop (energy jumps sharply
        right at the boundary) or a build (energy was climbing into it)."""
        frame_blocks = max(int(frame_seconds / block_duration), 1)
        num_frames = len(raw_band_energy) // frame_blocks
        if num_frames < kernel_radius_frames * 4:
            return []  # too short for meaningful structure analysis

        frames = raw_band_energy[:num_frames * frame_blocks].reshape(num_frames, frame_blocks, -1).mean(axis=1)
        frame_energy = frames.sum(axis=1)

        norms = np.linalg.norm(frames, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        similarity = (frames / norms) @ (frames / norms).T

        r = kernel_radius_frames
        axis = np.arange(-r, r)
        sign = np.where(axis < 0, -1.0, 1.0)
        gaussian = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * (r / 2) ** 2))
        kernel = np.outer(sign, sign) * gaussian

        novelty = np.zeros(num_frames)
        for t in range(r, num_frames - r):
            novelty[t] = np.sum(similarity[t - r:t + r, t - r:t + r] * kernel)

        threshold = np.mean(novelty) + np.std(novelty)
        min_gap_frames = max(int(min_gap_seconds / frame_seconds), 1)

        boundaries = []
        last_peak = -min_gap_frames
        for t in range(r, num_frames - r):
            if novelty[t] <= threshold:
                continue
            if novelty[t] < novelty[t - 1] or novelty[t] < novelty[t + 1]:
                continue  # not a local peak
            if t - last_peak < min_gap_frames:
                continue
            last_peak = t

            energy_before = np.mean(frame_energy[max(t - build_frames, 0):t])
            energy_after = np.mean(frame_energy[t:t + build_frames])
            if energy_before > 0 and energy_after >= energy_before * drop_ratio:
                boundary_type = "drop"
            elif t >= build_frames and np.all(np.diff(frame_energy[t - build_frames:t]) > 0):
                boundary_type = "build"
            else:
                boundary_type = "transition"

            boundaries.append({"time": round(t * frame_seconds, 2), "type": boundary_type})

        return boundaries

    def _store_features(self, song_id, features):
        with self.connection.cursor() as cur:
            cur.execute("""
                INSERT INTO song_features (song_id, features)
                VALUES (%s, %s)
                ON CONFLICT (song_id) DO UPDATE SET features = EXCLUDED.features, computed_at = now();
            """, (song_id, psycopg2.extras.Json(features)))
