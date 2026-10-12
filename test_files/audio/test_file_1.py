import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import matplotlib.animation as animation

from arlx.audio.sources.input_capture import InputAudioCapture

CHANNELS = 1
HISTORY_LENGTH_SECONDS = 1  # seconds
SAMPLES_BLOCK_SIZE = 1024
SAMPLE_RATE = 44100
BLOCK_SIZE_DURATION = SAMPLES_BLOCK_SIZE / SAMPLE_RATE  # seconds

HISTORY_LENGTH_BLOCKS = int(HISTORY_LENGTH_SECONDS / BLOCK_SIZE_DURATION)  # number of blocks to keep in history

raw_audio_deque = deque(maxlen=HISTORY_LENGTH_BLOCKS)
for _ in range(HISTORY_LENGTH_BLOCKS):
    raw_audio_deque.append(np.zeros((1024, CHANNELS)))

spec_x = np.fft.fftfreq(SAMPLES_BLOCK_SIZE, d=1.0 / SAMPLE_RATE)[:SAMPLES_BLOCK_SIZE // 2]
fft_data = np.zeros(SAMPLES_BLOCK_SIZE // 2)

fig, (ax1, ax2) = plt.subplots(2,3)
t = np.linspace(-HISTORY_LENGTH_SECONDS, 0, 1024*HISTORY_LENGTH_BLOCKS)

lines = []
for i in range(CHANNELS):
    line = ax1.plot(0, 0, label=f'Channel {i+1}')[0]
    lines.append(line)

fft_line = ax2.plot(spec_x, fft_data)[0]

ax1.set(xlim=(-HISTORY_LENGTH_SECONDS, 0), ylim=(-2, 2), xlabel='Time [s]', ylabel='Amplitude')
ax1.legend()

FREQUENCY_BANDS = [
    ("Sub-bass", 20, 60, "#4b0082"),
    ("Bass", 60, 250, "#0000ff"),
    ("Low Mid", 250, 500, "#00aaff"),
    ("Mid", 500, 2000, "#00c853"),
    ("High Mid", 2000, 4000, "#ffd600"),
    ("Presence", 4000, 6000, "#ff9100"),
    ("Brilliance", 6000, 20000, "#ff1744"),
]

ax2.set(xlim=(20, SAMPLE_RATE / 2), ylim=(0, 100), xlabel='Frequency [Hz]', ylabel='Magnitude')
ax2.set_xscale('log')

for name, low, high, color in FREQUENCY_BANDS:
    ax2.axvspan(low, high, color=color, alpha=0.2)
    ax2.text((low * high) ** 0.5, 95, name, rotation=90, ha='center', va='top', fontsize=7)

def update(frame):
    for i, line in enumerate(lines):
        line.set_data(t, np.concatenate([block[:, i] for block in raw_audio_deque], axis=0))
    fft_line.set_data(spec_x, fft_data)
    return frame

ani = animation.FuncAnimation(fig=fig, func=update, interval=30, cache_frame_data=False)
plt.show(block=False)


try:
    stream = InputAudioCapture(device_name="default", block_size=1024)
    stream.start()
    print("Press Ctrl+C to stop the stream...")
    while True:
        returned, latest_sample, timestamp = stream.get_latest_sample_with_timestamp()
        if latest_sample is not None:
            raw_audio_deque.append(latest_sample.copy())

            y = np.fft.fft(latest_sample[0:SAMPLES_BLOCK_SIZE, 0])
            fft_data = np.abs(y[:SAMPLES_BLOCK_SIZE // 2])

        plt.pause(0.01)
except KeyboardInterrupt:
    stream.stop()
    print("Stopping the stream...")
