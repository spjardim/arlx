import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sounddevice as sd

import dejavu.logic.decoder as decoder
from dejavu import Dejavu

from arlx.audio.song_library import SongLibrary
from arlx.audio.test_file_17 import REPO_ROOT, compute_envelope

# Plays a song start to finish while showing its *entire* precomputed
# structure up front (we already know the whole file, so there's no reason to
# only reveal boundaries as they're reached) - colored bands mark each
# detected drop/transition/build, and a moving playhead shows where real-time
# playback currently is relative to them.
WAVEFORM_COLOR = "#1976d2"
PLAYHEAD_COLOR = "#212121"
BOUNDARY_COLORS = {
    "drop": "#e53935",
    "build": "#fb8c00",
    "transition": "#5e35b1",
}
BOUNDARY_HALF_WIDTH_SECONDS = 1.5

config = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "password",
        "database": "dejavu",
    },
    "database_type": "postgres",
}

current_position_seconds = 0.0
stop_event = threading.Event()


def audio_worker(samples, fs, block_size):
    global current_position_seconds

    stream = sd.OutputStream(samplerate=fs, channels=1, dtype="float32", blocksize=block_size)
    stream.start()
    try:
        for start in range(0, len(samples), block_size):
            if stop_event.is_set():
                break
            block = samples[start:start + block_size]
            if len(block) < block_size:
                break
            stream.write(block)
            current_position_seconds = start / fs
    finally:
        stream.stop()
        stream.close()


def plot_with_structure(song_path, samples, fs, features):
    fig, ax = plt.subplots(figsize=(14, 5))

    t_env, envelope = compute_envelope(samples, fs)
    ax.fill_between(t_env, -envelope, envelope, color=WAVEFORM_COLOR, linewidth=0)
    ax.set(xlim=(0, features["duration_seconds"]), ylim=(-1, 1), xlabel="Time [s]", ylabel="Amplitude")
    ax.set_title(song_path.name)

    for boundary in features["structure"]:
        t = boundary["time"]
        color = BOUNDARY_COLORS.get(boundary["type"], "#9e9e9e")
        ax.axvspan(t - BOUNDARY_HALF_WIDTH_SECONDS, t + BOUNDARY_HALF_WIDTH_SECONDS, color=color, alpha=0.35)
        ax.text(t, 1.05, boundary["type"], color=color, ha="center", va="bottom",
                fontsize=9, rotation=0, clip_on=False)

    playhead = ax.axvline(0, color=PLAYHEAD_COLOR, linewidth=2)

    def update(frame):
        playhead.set_xdata([current_position_seconds, current_position_seconds])
        return (playhead,)

    ani = animation.FuncAnimation(fig=fig, func=update, interval=100, cache_frame_data=False)
    return fig, ani


if __name__ == "__main__":
    if len(sys.argv) > 1:
        song_path = Path(sys.argv[1])
    else:
        song_path = REPO_ROOT / "songs" / "runaway.flac"

    djv = Dejavu(config)
    library = SongLibrary(djv, config["database"])

    print(f"Loading/computing features for {song_path.name}...")
    song_id = library.add_song(str(song_path))
    features = library.get_features(song_id)
    print(f"  {features['duration_seconds']:.0f}s, {len(features['structure'])} structure boundaries")

    channels, fs, _ = decoder.read(str(song_path))
    samples = channels[0].astype(np.float32) / 32767.0

    fig, ani = plot_with_structure(song_path, samples, fs, features)

    worker_thread = threading.Thread(target=audio_worker, args=(samples, fs, 1024), daemon=True)
    worker_thread.start()

    plt.tight_layout()
    plt.show()

    stop_event.set()
    worker_thread.join(timeout=1)
