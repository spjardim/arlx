import sys
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from test_files.audio.test_file_17 import REPO_ROOT, analyze_song, compute_envelope

# Compares our own tier0-pipeline beat detection (test_file_17's analyze_song)
# against librosa's dynamic-programming beat tracker, on the same file. Not
# touching any of the tier0 classes - this is purely "which approach should
# the offline/precompute path actually use," per the low-confidence BPM result
# seen in test_file_17.
WAVEFORM_COLOR = "#1976d2"
OUR_BEAT_COLOR = "#9e9e9e"
LIBROSA_BEAT_COLOR = "#8e24aa"


def run_librosa(samples, fs):
    start = time.time()
    tempo, beat_times = librosa.beat.beat_track(y=samples, sr=fs, units="time")
    elapsed = time.time() - start
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    return tempo_value, beat_times, elapsed


def plot_comparison(song_path, data, librosa_tempo, librosa_beat_times, librosa_elapsed):
    fig, (ax_wave, ax_beats) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]},
    )

    t_env, envelope = compute_envelope(data["samples"], data["fs"])
    ax_wave.fill_between(t_env, -envelope, envelope, color=WAVEFORM_COLOR, linewidth=0)
    ax_wave.set(ylabel="Amplitude", ylim=(-1, 1))
    ax_wave.set_title(
        f"{song_path.name}  |  ours: {data['bpm_over_time'][-1]:.1f} BPM"
        f"  |  librosa: {librosa_tempo:.1f} BPM ({librosa_elapsed:.1f}s)"
    )

    ax_beats.eventplot([data["beat_times"]], lineoffsets=0.75, linelengths=0.4, colors=OUR_BEAT_COLOR)
    ax_beats.eventplot([librosa_beat_times], lineoffsets=0.25, linelengths=0.4, colors=LIBROSA_BEAT_COLOR)
    ax_beats.set(
        ylabel="Beats",
        xlabel="Time [s]",
        yticks=[0.25, 0.75],
        yticklabels=["librosa", "ours"],
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        song_path = Path(sys.argv[1])
    else:
        song_path = sorted((REPO_ROOT / "songs").glob("*.flac"))[0]

    print(f"Analyzing {song_path.name} with our pipeline...")
    data = analyze_song(song_path)
    print(f"  ours: {data['bpm_over_time'][-1]:.1f} BPM, {len(data['beat_times'])} beats")

    print("Analyzing with librosa...")
    librosa_tempo, librosa_beat_times, librosa_elapsed = run_librosa(data["samples"], data["fs"])
    print(f"  librosa: {librosa_tempo:.1f} BPM, {len(librosa_beat_times)} beats ({librosa_elapsed:.1f}s)")

    plot_comparison(song_path, data, librosa_tempo, librosa_beat_times, librosa_elapsed)
