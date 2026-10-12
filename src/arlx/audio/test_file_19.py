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

# Live/real-time test of SongDetector's cheap sync-check (added after
# test_file_16): plays songs through speakers while feeding the same
# unaltered blocks into SongDetector, same as test_file_16, but now also
# tracks sync confidence over time and counts real dejavu calls, so we can
# confirm - under real wall-clock timing, not an instant bash loop - that
# dejavu only fires once per song instead of repeatedly re-matching.
REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
RECORD_SECONDS = 5
HISTORY_SECONDS = 1
SYNC_HISTORY_SECONDS = 15

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

sync_confidence_history = deque(maxlen=int(SYNC_HISTORY_SECONDS / (BLOCK_SIZE / SAMPLE_RATE)))
for _ in range(sync_confidence_history.maxlen):
    sync_confidence_history.append(0.0)
sync_time_axis = np.linspace(-SYNC_HISTORY_SECONDS, 0, sync_confidence_history.maxlen)

dejavu_call_count = {"n": 0}
stop_event = threading.Event()
song_detector = None  # created in audio_worker, read by the plot


def audio_worker():
    global song_detector

    djv = Dejavu(config)

    # Wrap generate_fingerprints just to count real dejavu calls for display -
    # doesn't change behavior, only observes it.
    real_generate = djv.generate_fingerprints

    def counting_generate(*args, **kwargs):
        dejavu_call_count["n"] += 1
        return real_generate(*args, **kwargs)

    djv.generate_fingerprints = counting_generate

    song_detector = SongDetector(djv, record_seconds=RECORD_SECONDS, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE,
                                  detect_every_seconds=1.0, sync_check_seconds=1.0, sync_check_every_seconds=0.5)
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
                sync_confidence_history.append(song_detector.get_sync_confidence())
        finally:
            stream.stop()
            stream.close()


fig, (ax_wave, ax_sync) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.5, 1]})
t_blue = np.linspace(-HISTORY_SECONDS, 0, raw_audio_deque.maxlen * BLOCK_SIZE)
blue_line = ax_wave.plot(t_blue, np.zeros_like(t_blue), color="#1976d2", label="live/played")[0]
orange_line = ax_wave.plot([], [], color="#fb8c00", label="reference (known song)")[0]
ax_wave.set(xlim=(-HISTORY_SECONDS, HISTORY_SECONDS), ylim=(-1.2, 1.2), xlabel="Time [s]", ylabel="Amplitude")
ax_wave.axvline(0, color="black", linewidth=0.5)
ax_wave.legend(loc="upper left")

status_text = ax_wave.text(0.99, 0.97, "", transform=ax_wave.transAxes, ha="right", va="top", fontsize=10,
                            family="monospace")

sync_line = ax_sync.plot(sync_time_axis, list(sync_confidence_history), color="#00897b")[0]
ax_sync.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="fallback threshold")
ax_sync.set(xlim=(-SYNC_HISTORY_SECONDS, 0), ylim=(0, 1.05), xlabel="Time [s]", ylabel="Sync confidence")
ax_sync.legend(loc="lower left")


def update(frame):
    blue_line.set_ydata(np.concatenate(list(raw_audio_deque)))

    if song_detector is not None:
        t_orange, window = song_detector.get_reference_window(HISTORY_SECONDS)
        if window is not None:
            orange_line.set_data(t_orange, window)
        else:
            orange_line.set_data([], [])

        position = song_detector.get_current_position_seconds()
        position_text = f"{position:.2f}s" if position is not None else "--"
        status_text.set_text(
            f"song: {song_detector.get_song_name()}\n"
            f"position: {position_text}\n"
            f"sync confidence: {song_detector.get_sync_confidence():.2f}\n"
            f"dejavu calls so far: {dejavu_call_count['n']}"
        )

    sync_line.set_data(sync_time_axis, list(sync_confidence_history))

    return blue_line, orange_line, sync_line, status_text


ani = animation.FuncAnimation(fig=fig, func=update, interval=30, cache_frame_data=False)

worker_thread = threading.Thread(target=audio_worker, daemon=True)
worker_thread.start()

plt.tight_layout()
plt.show()

stop_event.set()
worker_thread.join(timeout=1)
