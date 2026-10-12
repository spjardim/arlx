import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dejavu.logic.decoder as decoder

from arlx.audio.fft_analyzer import FFTAnalyzer
from arlx.audio.spectral_flux import SpectralFlux
from arlx.audio.onest_detection import OnsetDetection
from arlx.audio.tempo_tracker import TempoTracker
from arlx.audio.beat_phase_tracker import BeatPhaseTracker
from arlx.audio.downbeat_detector import DownbeatDetector

# Offline batch run of the whole tier0 pipeline over a known file (no mic, no
# threading needed - just reading a file we already have start to finish).
# This is the "precompute everything once when a song is added" step from the
# SongLibrary discussion, made visible: one full breakdown of a song, so we
# can see what we'd actually be storing before building the real cache.
REPO_ROOT = Path(__file__).resolve().parents[3]

BLOCK_SIZE = 1024

WAVEFORM_COLOR = "#1976d2"
ONSET_COLOR = "#ff1744"
BEAT_COLOR = "#9e9e9e"
DOWNBEAT_COLOR = "#ffc400"
BPM_COLOR = "#00897b"


def compute_envelope(samples, fs, resolution_seconds=0.1):
    window = int(resolution_seconds * fs)
    n_windows = len(samples) // window
    trimmed = samples[:n_windows * window].reshape(n_windows, window)
    envelope = np.max(np.abs(trimmed), axis=1)
    t = np.arange(n_windows) * resolution_seconds
    return t, envelope


def analyze_song(file_path):
    channels, fs, _ = decoder.read(str(file_path))
    samples = channels[0].astype(np.float32) / 32767.0
    block_duration = BLOCK_SIZE / fs

    fft_analyzer = FFTAnalyzer(sample_rate=fs, block_size=BLOCK_SIZE)
    spectral_flux = SpectralFlux(sample_rate=fs, frame_size=BLOCK_SIZE, hop_size=BLOCK_SIZE)
    onset_detection = OnsetDetection(sample_rate=fs, hop_length=BLOCK_SIZE)
    tempo_tracker = TempoTracker(sample_rate=fs, hop_length=BLOCK_SIZE)
    beat_phase_tracker = BeatPhaseTracker(sample_rate=fs, hop_length=BLOCK_SIZE)
    downbeat_detector = DownbeatDetector()

    num_blocks = len(samples) // BLOCK_SIZE

    band_energy_over_time = np.zeros((num_blocks, 7))
    bpm_over_time = np.zeros(num_blocks)
    confidence_over_time = np.zeros(num_blocks)
    onset_times = []
    beat_times = []
    downbeat_times = []

    for i in range(num_blocks):
        block = samples[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
        t = i * block_duration

        fft_analyzer.process(block)
        spectral_flux.process(fft_analyzer.get_raw_fft_data())
        onset_detection.process(spectral_flux.get_global_flux(), spectral_flux.get_band_fluxes())

        band_onsets = onset_detection.get_band_onsets()
        kick_onset = bool(band_onsets[0] or band_onsets[1])  # Sub-bass or Bass band

        tempo_tracker.process(kick_onset)
        beat_phase_tracker.process(kick_onset, tempo_tracker.get_bpm())
        downbeat_detector.process(beat_phase_tracker.get_beat_now(), spectral_flux.get_global_flux())

        band_energy_over_time[i] = fft_analyzer.get_smoothed_norm_bins()
        bpm_over_time[i] = tempo_tracker.get_bpm()
        confidence_over_time[i] = tempo_tracker.get_confidence()

        if kick_onset:
            onset_times.append(t)
        if beat_phase_tracker.get_beat_now():
            beat_times.append(t)
            if downbeat_detector.get_is_downbeat():
                downbeat_times.append(t)

    return {
        "samples": samples,
        "fs": fs,
        "block_duration": block_duration,
        "num_blocks": num_blocks,
        "band_energy_over_time": band_energy_over_time,
        "bpm_over_time": bpm_over_time,
        "confidence_over_time": confidence_over_time,
        "onset_times": onset_times,
        "beat_times": beat_times,
        "downbeat_times": downbeat_times,
    }


def plot_breakdown(song_path, data):
    fig, (ax_wave, ax_bands, ax_bpm, ax_confidence, ax_beats) = plt.subplots(
        5, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.5, 1, 0.4, 0.5]},
    )

    t_env, envelope = compute_envelope(data["samples"], data["fs"])
    ax_wave.fill_between(t_env, -envelope, envelope, color=WAVEFORM_COLOR, linewidth=0)
    ax_wave.set(ylabel="Amplitude", ylim=(-1, 1))
    ax_wave.set_title(song_path.name)

    block_times = np.arange(data["num_blocks"]) * data["block_duration"]

    ax_bands.imshow(
        data["band_energy_over_time"].T,
        aspect="auto",
        origin="lower",
        extent=[block_times[0], block_times[-1], 0, 7],
        cmap="cividis",
        vmin=0, vmax=1,
    )
    ax_bands.set(ylabel="Band\n(low -> high)")

    ax_bpm.plot(block_times, data["bpm_over_time"], color=BPM_COLOR)
    ax_bpm.set(ylabel="BPM", ylim=(60, 190))

    ax_confidence.fill_between(block_times, 0, data["confidence_over_time"], color=BPM_COLOR, alpha=0.5, linewidth=0)
    ax_confidence.set(ylabel="Confidence", ylim=(0, 1))

    ax_beats.eventplot([data["beat_times"]], lineoffsets=0.7, linelengths=0.5, colors=BEAT_COLOR)
    ax_beats.eventplot([data["downbeat_times"]], lineoffsets=0.3, linelengths=0.5,
                        colors=DOWNBEAT_COLOR, linewidths=2)
    ax_beats.set(ylabel="Beats", xlabel="Time [s]", yticks=[])

    for onset_t in data["onset_times"]:
        ax_wave.axvline(onset_t, color=ONSET_COLOR, alpha=0.1, linewidth=0.8, zorder=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        song_path = Path(sys.argv[1])
    else:
        song_path = sorted((REPO_ROOT / "songs").glob("*.flac"))[0]

    print(f"Analyzing {song_path.name}...")
    data = analyze_song(song_path)
    print(
        f"{data['num_blocks'] * data['block_duration']:.1f}s, "
        f"{len(data['onset_times'])} onsets, {len(data['beat_times'])} beats, "
        f"{len(data['downbeat_times'])} downbeats, final BPM: {data['bpm_over_time'][-1]:.1f}"
    )
    plot_breakdown(song_path, data)
