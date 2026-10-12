import threading
from collections import deque
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

import dejavu.logic.decoder as decoder
from dejavu import Dejavu

from arlx.audio.song_detector import SongDetector

# Progression from test_file_15: same plot (live audio on [-1,0] in blue vs.
# the known reference file on [-1,+1] in orange), but now all the position
# tracking, detection throttling, and reference loading lives inside
# SongDetector itself via process()/get_current_position_seconds()/
# get_reference_window() - this script just plays audio and reads state.
REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
RECORD_SECONDS = 5
HISTORY_SECONDS = 1

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

raw_audio_deque = deque(maxlen=int(HISTORY_SECONDS / (BLOCK_SIZE / SAMPLE_RATE)))
for _ in range(raw_audio_deque.maxlen):
    raw_audio_deque.append(np.zeros(BLOCK_SIZE, dtype=np.float32))

stop_event = threading.Event()
song_detector = None  # created in audio_worker, read by the plot


def audio_worker():
    global song_detector

    djv = Dejavu(config)
    song_detector = SongDetector(djv, record_seconds=RECORD_SECONDS, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, detect_every_seconds=5)
    song_detector.fingerprint_directory(str(REPO_ROOT / "songs"), [".flac"])

    for song_path in sorted((REPO_ROOT / "songs").glob("*.flac")):
        if stop_event.is_set():
            break

        channels, fs, _ = decoder.read(str(song_path))
        samples_float32 = channels[0].astype(np.float32) / 32767.0

        print(f"\nPlaying {song_path.name}...")
        stream = sd.OutputStream(samplerate=fs, channels=1, dtype="float32", blocksize=BLOCK_SIZE)
        stream.start()
        try:
            for start in range(0, len(samples_float32), BLOCK_SIZE):
                if stop_event.is_set():
                    break

                block = samples_float32[start:start + BLOCK_SIZE]
                if len(block) < BLOCK_SIZE:
                    break

                stream.write(block)
                song_detector.process(block)
                raw_audio_deque.append(block)
        finally:
            stream.stop()
            stream.close()


fig, ax = plt.subplots()
t_blue = np.linspace(-HISTORY_SECONDS, 0, raw_audio_deque.maxlen * BLOCK_SIZE)
blue_line = ax.plot(t_blue, np.zeros_like(t_blue), color="#1976d2", label="live/played")[0]
orange_line = ax.plot([], [], color="#fb8c00", label="reference (known song)")[0]
ax.set(xlim=(-HISTORY_SECONDS, HISTORY_SECONDS), ylim=(-1.2, 1.2), xlabel="Time [s]", ylabel="Amplitude")
ax.axvline(0, color="black", linewidth=0.5)
ax.legend()


def update(frame):
    blue_line.set_ydata(np.concatenate(list(raw_audio_deque)))

    if song_detector is not None:
        t_orange, window = song_detector.get_reference_window(HISTORY_SECONDS)
        if window is not None:
            orange_line.set_data(t_orange, window)
        else:
            orange_line.set_data([], [])

    return blue_line, orange_line


ani = animation.FuncAnimation(fig=fig, func=update, interval=30, cache_frame_data=False)

worker_thread = threading.Thread(target=audio_worker, daemon=True)
worker_thread.start()

plt.show()

stop_event.set()
worker_thread.join(timeout=1)
