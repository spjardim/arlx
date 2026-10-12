import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

import dejavu.logic.decoder as decoder
from dejavu import Dejavu

from arlx.audio.song_detector import SongDetector

# Progression from test_file_14: same play-and-feed-clean-samples setup, but
# now also tracks *where in the song* we are, with the raw math inline here
# rather than a new class (fold into SongDetector later once this is proven
# out). Plots the live/played audio on [-1, 0] (blue) against the matching
# slice of the known reference file on [-1, +1] (orange) - if our position
# estimate is right, blue and orange should overlap almost exactly on the
# shared [-1, 0] half, and orange alone shows into [0, +1] - audio we haven't
# "heard" yet but already know because we've identified the song.
REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
BLOCK_DURATION = BLOCK_SIZE / SAMPLE_RATE
RECORD_SECONDS = 5
HISTORY_SECONDS = 1

DETECT_EVERY_SECONDS = 1.0
DETECT_EVERY_BLOCKS = int(DETECT_EVERY_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
HISTORY_BLOCKS = int(HISTORY_SECONDS / BLOCK_DURATION)

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

# --- shared state between the audio worker thread and the plot ---
raw_audio_deque = deque(maxlen=HISTORY_BLOCKS)
for _ in range(HISTORY_BLOCKS):
    raw_audio_deque.append(np.zeros(BLOCK_SIZE, dtype=np.float32))

elapsed_time = 0.0
estimated_position_seconds = None  # where we think we are in the reference track, as of the last detection
position_anchor_elapsed_time = None  # elapsed_time when that estimate was made - lets us extrapolate forward

current_song_samples = None  # full reference track, loaded once it's identified
current_song_fs = None
current_song_name = None

stop_event = threading.Event()


def audio_worker():
    global elapsed_time, estimated_position_seconds, position_anchor_elapsed_time
    global current_song_samples, current_song_fs, current_song_name

    djv = Dejavu(config)
    song_detector = SongDetector(djv, record_seconds=RECORD_SECONDS, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
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
            block_count = 0
            for start in range(0, len(samples_float32), BLOCK_SIZE):
                if stop_event.is_set():
                    break

                block = samples_float32[start:start + BLOCK_SIZE]
                if len(block) < BLOCK_SIZE:
                    break

                stream.write(block)
                song_detector.add_samples(block)
                raw_audio_deque.append(block)
                elapsed_time += BLOCK_DURATION
                block_count += 1

                if block_count % DETECT_EVERY_BLOCKS == 0:
                    results = song_detector.detect_song_from_blocks()
                    if results:
                        top = results[0]
                        song_name = top["song_name"].decode()
                        print(f"Heard: {song_name} (offset {top['offset_seconds']}s)")

                        # Load the reference file (if it's a new song) BEFORE
                        # publishing the position estimate below - otherwise
                        # the plot thread can see a non-None position with no
                        # reference data loaded yet and crash.
                        if song_name != current_song_name:
                            ref_path = REPO_ROOT / "songs" / f"{song_name}.flac"
                            ref_channels, ref_fs, _ = decoder.read(str(ref_path))
                            current_song_samples = ref_channels[0].astype(np.float32) / 32767.0
                            current_song_fs = ref_fs
                            current_song_name = song_name

                        # offset_seconds is where the START of the fingerprinted
                        # window (the last RECORD_SECONDS of audio) sits in the
                        # reference track - "now" is the END of that window.
                        estimated_position_seconds = top["offset_seconds"] + RECORD_SECONDS
                        position_anchor_elapsed_time = elapsed_time
        finally:
            stream.stop()
            stream.close()


def get_reference_window():
    """Slice of the known reference file spanning [-1, +1] around our
    current estimated position. None if we haven't identified a song yet."""
    if estimated_position_seconds is None:
        return None, None

    current_position = estimated_position_seconds + (elapsed_time - position_anchor_elapsed_time)
    start_sample = int((current_position - HISTORY_SECONDS) * current_song_fs)
    end_sample = int((current_position + HISTORY_SECONDS) * current_song_fs)

    start_sample_clamped = max(start_sample, 0)
    end_sample_clamped = min(end_sample, len(current_song_samples))
    if start_sample_clamped >= end_sample_clamped:
        return None, None

    window = current_song_samples[start_sample_clamped:end_sample_clamped]
    t = np.linspace(
        (start_sample_clamped - start_sample) / current_song_fs - HISTORY_SECONDS,
        HISTORY_SECONDS - (end_sample - end_sample_clamped) / current_song_fs,
        len(window),
    )
    return t, window


fig, ax = plt.subplots()
t_blue = np.linspace(-HISTORY_SECONDS, 0, HISTORY_BLOCKS * BLOCK_SIZE)
blue_line = ax.plot(t_blue, np.zeros_like(t_blue), color="#1976d2", label="live/played")[0]
orange_line = ax.plot([], [], color="#fb8c00", label="reference (known song)")[0]
ax.set(xlim=(-HISTORY_SECONDS, HISTORY_SECONDS), ylim=(-1.2, 1.2), xlabel="Time [s]", ylabel="Amplitude")
ax.axvline(0, color="black", linewidth=0.5)
ax.legend()


def update(frame):
    blue_line.set_ydata(np.concatenate(list(raw_audio_deque)))

    t_orange, window = get_reference_window()
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
