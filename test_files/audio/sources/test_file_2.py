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

deque_data = deque(maxlen=HISTORY_LENGTH_BLOCKS)
for _ in range(HISTORY_LENGTH_BLOCKS):
    deque_data.append(np.zeros((1024, CHANNELS)))


fig, ax = plt.subplots()
t = np.linspace(-HISTORY_LENGTH_SECONDS, 0, 1024*HISTORY_LENGTH_BLOCKS)

lines = []
for i in range(CHANNELS):
    line = ax.plot(0, 0, label=f'Channel {i+1}')[0]
    lines.append(line)

ax.set(xlim=(-HISTORY_LENGTH_SECONDS, 0), ylim=(-2, 2), xlabel='Time [s]', ylabel='Amplitude')
ax.legend()

def update(frame):
    for i, line in enumerate(lines):
        line.set_data(t, np.concatenate([block[:, i] for block in deque_data], axis=0))
    return frame

ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
plt.show(block=False)


try:
    stream = InputAudioCapture(device_name="default", block_size=1024)
    stream.start()
    print("Press Ctrl+C to stop the stream...")
    while True:
        returned, latest_sample, timestamp = stream.get_latest_sample_with_timestamp()
        if latest_sample is not None:
            deque_data.append(latest_sample.copy())
        plt.pause(0.01)
except KeyboardInterrupt:
    stream.stop()
    print("Stopping the stream...")
