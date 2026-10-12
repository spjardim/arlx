import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import matplotlib.animation as animation

from arlx.audio.sources.input_capture import InputAudioCapture
from arlx.audio.fft_analyzer import FFTAnalyzer
HISTORY_LENGTH_SECONDS = 1  # seconds
SAMPLES_BLOCK_SIZE = 1024
SAMPLE_RATE = 44100
BLOCK_SIZE_DURATION = SAMPLES_BLOCK_SIZE / SAMPLE_RATE  # seconds

HISTORY_LENGTH_BLOCKS = int(HISTORY_LENGTH_SECONDS / BLOCK_SIZE_DURATION)  # number of blocks to keep in history

raw_audio_deque = deque(maxlen=HISTORY_LENGTH_BLOCKS)
for _ in range(HISTORY_LENGTH_BLOCKS):
    raw_audio_deque.append(np.zeros(1024))

fft_analyzer = FFTAnalyzer(
    sample_rate=SAMPLE_RATE,
    block_size=SAMPLES_BLOCK_SIZE,
)

spec_x = np.fft.fftfreq(SAMPLES_BLOCK_SIZE, d=1.0 / SAMPLE_RATE)[:SAMPLES_BLOCK_SIZE // 2]



fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3)
ax8.axis('off')
ax9.axis('off')
t = np.linspace(-HISTORY_LENGTH_SECONDS, 0, 1024*HISTORY_LENGTH_BLOCKS)

line = ax1.plot(0, 0, label=f'Channel 1')[0]

fft_line = ax2.plot(spec_x, fft_analyzer.get_raw_fft_data())[0]

fft_mean_bands_line = ax4.bar(range(7), fft_analyzer.get_mean_fft_bins(), color='blue', alpha=0.5)

fft_rms_bands_line = ax5.bar(range(7), fft_analyzer.get_rms_fft_bins(), color='green', alpha=0.5)

fft_rms_log_bands_line = ax6.bar(range(7), fft_analyzer.get_log_rms_fft_bins(), color='green', alpha=0.5)

fft_norm_bands_line = ax3.bar(range(7), fft_analyzer.get_norm_fft_bins(), color='orange', alpha=0.5)

fft_smoothed_bands_line = ax7.bar(range(7), fft_analyzer.get_smoothed_norm_bins(), color='red', alpha=0.5)

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

ax2.set()

ax4.set(xlim=(-0.5, 6.5), ylim=(0, 100), xlabel='Mean Frequency Bands', ylabel='Magnitude')
ax4.set_xticks(range(7))
ax4.set_xticklabels([name for name, _, _, _ in FREQUENCY_BANDS])

ax5.set(xlim=(-0.5, 6.5), ylim=(0, 100), xlabel='RMS Frequency Bands', ylabel='Magnitude')
ax5.set_xticks(range(7))
ax5.set_xticklabels([name for name, _, _, _ in FREQUENCY_BANDS])

ax6.set(xlim=(-0.5, 6.5), ylim=(-60, 40), xlabel='RMS Frequency Bands (dB)', ylabel='Magnitude (dB)')
ax6.set_xticks(range(7))
ax6.set_xticklabels([name for name, _, _, _ in FREQUENCY_BANDS])

ax3.set(xlim=(-0.5, 6.5), ylim=(0, 1), xlabel='Auto-Gain Normalized Bands', ylabel='Level (0-1)')
ax3.set_xticks(range(7))
ax3.set_xticklabels([name for name, _, _, _ in FREQUENCY_BANDS])

ax7.set(xlim=(-0.5, 6.5), ylim=(0, 1), xlabel='Temporally Smoothed Bands', ylabel='Level (0-1)')
ax7.set_xticks(range(7))
ax7.set_xticklabels([name for name, _, _, _ in FREQUENCY_BANDS])

def update(frame):
    line.set_data(t, np.concatenate([block for block in raw_audio_deque], axis=0))
    fft_line.set_data(spec_x, fft_analyzer.get_raw_fft_data())
    for rect, height in zip(fft_mean_bands_line, fft_analyzer.get_mean_fft_bins()):
        rect.set_height(height)
    for rect, height in zip(fft_rms_bands_line, fft_analyzer.get_rms_fft_bins()):
        rect.set_height(height)
    for rect, height in zip(fft_rms_log_bands_line, fft_analyzer.get_log_rms_fft_bins()):
        rect.set_height(height)
    for rect, height in zip(fft_norm_bands_line, fft_analyzer.get_norm_fft_bins()):
        rect.set_height(height)
    for rect, height in zip(fft_smoothed_bands_line, fft_analyzer.get_smoothed_norm_bins()):
        rect.set_height(height)

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
            raw_audio_deque.append(latest_sample[:, 0].copy())
            fft_analyzer.process(latest_sample[:, 0].copy())
        plt.pause(0.01)
except KeyboardInterrupt:
    stream.stop()
    print("Stopping the stream...")
