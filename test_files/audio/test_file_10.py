import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import matplotlib.animation as animation

from arlx.audio.sources.input_capture import InputAudioCapture
from arlx.audio.fft_analyzer import FFTAnalyzer
from arlx.audio.spectral_flux import SpectralFlux
from arlx.audio.onest_detection import OnsetDetection
from arlx.audio.tempo_tracker import TempoTracker
from arlx.audio.beat_phase_tracker import BeatPhaseTracker
from arlx.audio.downbeat_detector import DownbeatDetector

# Progression from test_file_9: added DownbeatDetector on top of the beat
# clock. ax15 shows the learned average strength at each position in the bar
# (gold = detected downbeat position, outlined = current position), ax16
# flashes "DOWNBEAT" when the current beat lands on that position.
HISTORY_LENGTH_SECONDS = 1  # seconds
SAMPLES_BLOCK_SIZE = 1024
SAMPLE_RATE = 44100
BLOCK_SIZE_DURATION = SAMPLES_BLOCK_SIZE / SAMPLE_RATE  # seconds

HISTORY_LENGTH_BLOCKS = int(HISTORY_LENGTH_SECONDS / BLOCK_SIZE_DURATION)  # number of blocks to keep in history
BPM_HISTORY_BLOCKS = int(10 / BLOCK_SIZE_DURATION)  # 10 seconds of smoothed BPM history

raw_audio_deque = deque(maxlen=HISTORY_LENGTH_BLOCKS)
for _ in range(HISTORY_LENGTH_BLOCKS):
    raw_audio_deque.append(np.zeros(1024))

fft_analyzer = FFTAnalyzer(
    sample_rate=SAMPLE_RATE,
    block_size=SAMPLES_BLOCK_SIZE,
)
spectral_flux = SpectralFlux(sample_rate=SAMPLE_RATE, frame_size=SAMPLES_BLOCK_SIZE, hop_size=SAMPLES_BLOCK_SIZE)
onset_detection = OnsetDetection(sample_rate=SAMPLE_RATE, hop_length=SAMPLES_BLOCK_SIZE)
tempo_tracker = TempoTracker(sample_rate=SAMPLE_RATE, hop_length=SAMPLES_BLOCK_SIZE)
beat_phase_tracker = BeatPhaseTracker(sample_rate=SAMPLE_RATE, hop_length=SAMPLES_BLOCK_SIZE)
downbeat_detector = DownbeatDetector()

bpm_history = deque(maxlen=BPM_HISTORY_BLOCKS)
for _ in range(BPM_HISTORY_BLOCKS):
    bpm_history.append(0.0)
bpm_time_axis = np.linspace(-10, 0, BPM_HISTORY_BLOCKS)

spec_x = np.fft.fftfreq(SAMPLES_BLOCK_SIZE, d=1.0 / SAMPLE_RATE)[:SAMPLES_BLOCK_SIZE // 2]

ONSET_COLOR = "#ff1744"
FLUX_COLOR = "purple"
BPM_COLOR = "#00897b"
BEAT_COLOR = "#ff1744"
PHASE_COLOR = "#3949ab"
DOWNBEAT_COLOR = "#ffc400"
POSITION_COLOR = "#78909c"

fig, ((ax1, ax2, ax3, ax10), (ax4, ax5, ax6, ax11), (ax7, ax8, ax9, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4)
t = np.linspace(-HISTORY_LENGTH_SECONDS, 0, 1024*HISTORY_LENGTH_BLOCKS)

line = ax1.plot(0, 0, label=f'Channel 1')[0]

fft_line = ax2.plot(spec_x, fft_analyzer.get_raw_fft_data())[0]

fft_mean_bands_line = ax4.bar(range(7), fft_analyzer.get_mean_fft_bins(), color='blue', alpha=0.5)

fft_rms_bands_line = ax5.bar(range(7), fft_analyzer.get_rms_fft_bins(), color='green', alpha=0.5)

fft_rms_log_bands_line = ax6.bar(range(7), fft_analyzer.get_log_rms_fft_bins(), color='green', alpha=0.5)

fft_norm_bands_line = ax3.bar(range(7), fft_analyzer.get_norm_fft_bins(), color='orange', alpha=0.5)

fft_smoothed_bands_line = ax7.bar(range(7), fft_analyzer.get_smoothed_norm_bins(), color='red',    alpha=0.5)

global_flux_line =        ax8.bar(0,        spectral_flux.get_global_flux(),       color=FLUX_COLOR, alpha=0.5)
global_threshold_line = ax8.axhline(onset_detection.get_global_threshold(), color='black', linestyle='--')

local_flux_line = ax9.bar(range(7), spectral_flux.get_band_fluxes(), color=FLUX_COLOR, alpha=0.5)
local_threshold_line = ax9.plot(range(7), onset_detection.get_band_thresholds(), 'k--', marker='o')[0]

autocorrelation_line = ax10.plot(tempo_tracker.get_bpm_axis(), tempo_tracker.get_autocorrelation())[0]
autocorrelation_peak_line = ax10.axvline(0, color=BPM_COLOR, linestyle='--')

bpm_history_line = ax11.plot(bpm_time_axis, list(bpm_history), color=BPM_COLOR)[0]

bpm_text = ax12.text(0.5, 0.5, "BPM: --\nConfidence: --", ha='center', va='center', fontsize=14)
ax12.set_axis_off()

phase_bar = ax13.barh(0, 0.0, color=PHASE_COLOR)[0]

beat_text = ax14.text(0.5, 0.5, "next beat in --s", ha='center', va='center', fontsize=14)
ax14.set_axis_off()

position_bars = ax15.bar(range(downbeat_detector.beats_per_bar), downbeat_detector.get_position_strengths(),
                          color=POSITION_COLOR)
position_markers = ax15.plot([0], [0], 'v', color='black', markersize=12)[0]

downbeat_text = ax16.text(0.5, 0.5, "beat 1 / 4", ha='center', va='center', fontsize=14)
ax16.set_axis_off()

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

ax8.set(xlim=(-0.5, 0.5), ylim=(0, 100), xlabel='Global Spectral Flux + Onset', ylabel='Magnitude')
ax8.set_xticks([0])
ax8.set_xticklabels(['Global'])
ax9.set(xlim=(-0.5, 6.5), ylim=(0, 100), xlabel='Local Spectral Flux + Onset', ylabel='Magnitude')
ax9.set_xticks(range(7))
ax9.set_xticklabels([name for name, _, _, _ in FREQUENCY_BANDS])

ax10.set(xlim=(tempo_tracker.min_bpm, tempo_tracker.max_bpm), ylim=(0, 1),
          xlabel='BPM candidate', ylabel='Autocorrelation (normalized)')
ax11.set(xlim=(-10, 0), ylim=(tempo_tracker.min_bpm, tempo_tracker.max_bpm),
          xlabel='Time [s]', ylabel='Smoothed BPM')

ax13.set(xlim=(0, 1), ylim=(-0.5, 0.5), xlabel='Beat Phase (0 = beat, 1 = next beat)')
ax13.set_yticks([])

ax15.set(xlim=(-0.5, downbeat_detector.beats_per_bar - 0.5), ylim=(0, 100),
          xlabel='Position in Bar', ylabel='Avg Strength')
ax15.set_xticks(range(downbeat_detector.beats_per_bar))

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

    for rect, height in zip(local_flux_line, spectral_flux.get_band_fluxes()):
        rect.set_height(height)
    global_flux_line[0].set_height(spectral_flux.get_global_flux())

    global_flux_line[0].set_color(ONSET_COLOR if onset_detection.get_global_onset() else FLUX_COLOR)
    for rect, onset in zip(local_flux_line, onset_detection.get_band_onsets()):
        rect.set_color(ONSET_COLOR if onset else FLUX_COLOR)

    global_threshold_line.set_ydata([onset_detection.get_global_threshold()] * 2)
    local_threshold_line.set_ydata(onset_detection.get_band_thresholds())

    autocorrelation = tempo_tracker.get_autocorrelation()
    autocorrelation_max = autocorrelation.max() if autocorrelation.max() > 0 else 1.0
    autocorrelation_line.set_data(tempo_tracker.get_bpm_axis(), autocorrelation / autocorrelation_max)
    autocorrelation_peak_line.set_xdata([tempo_tracker.get_bpm(), tempo_tracker.get_bpm()])

    bpm_history_line.set_data(bpm_time_axis, list(bpm_history))

    bpm_text.set_text(f"BPM: {tempo_tracker.get_bpm():.1f}\nConfidence: {tempo_tracker.get_confidence():.0%}")

    phase_bar.set_width(beat_phase_tracker.get_phase())
    phase_bar.set_color(BEAT_COLOR if beat_phase_tracker.get_beat_now() else PHASE_COLOR)

    beat_text.set_text(f"next beat in {beat_phase_tracker.get_next_beat_in():.2f}s")
    beat_text.set_color(BEAT_COLOR if beat_phase_tracker.get_beat_now() else "black")

    for i, (rect, height) in enumerate(zip(position_bars, downbeat_detector.get_position_strengths())):
        rect.set_height(height)
        rect.set_color(DOWNBEAT_COLOR if i == downbeat_detector.get_downbeat_offset() else POSITION_COLOR)
    position_markers.set_data([downbeat_detector.get_beat_index()], [95])

    downbeat_text.set_text(
        f"beat {downbeat_detector.get_beat_index() + 1} / {downbeat_detector.beats_per_bar}" +
        ("\nDOWNBEAT" if downbeat_detector.get_is_downbeat() else "")
    )
    downbeat_text.set_color(DOWNBEAT_COLOR if downbeat_detector.get_is_downbeat() else "black")

    return frame

ani = animation.FuncAnimation(fig=fig, func=update, interval=30, cache_frame_data=False)

# Audio capture + the whole analysis chain run on their own thread so a slow
# redraw (many subplots) can never throttle how often real blocks get
# consumed - matplotlib's FuncAnimation timer runs independently in the main
# thread and just reads whatever get_*() returns, whenever it gets a turn.
stop_event = threading.Event()


def audio_worker():
    stream = InputAudioCapture(device_name="default", block_size=1024)
    stream.start()
    try:
        while not stop_event.is_set():
            returned, latest_sample, timestamp = stream.get_latest_sample_with_timestamp()
            if latest_sample is not None:
                raw_audio_deque.append(latest_sample[:, 0].copy())
                fft_analyzer.process(latest_sample[:, 0].copy())
                spectral_flux.process(fft_analyzer.get_raw_fft_data())
                onset_detection.process(spectral_flux.get_global_flux(), spectral_flux.get_band_fluxes())

                band_onsets = onset_detection.get_band_onsets()
                kick_onset = bool(band_onsets[0] or band_onsets[1])  # Sub-bass or Bass band
                tempo_tracker.process(kick_onset)
                bpm_history.append(tempo_tracker.get_bpm())

                beat_phase_tracker.process(kick_onset, tempo_tracker.get_bpm())
                downbeat_detector.process(beat_phase_tracker.get_beat_now(), spectral_flux.get_global_flux())
            else:
                time.sleep(0.001)
    finally:
        stream.stop()


worker_thread = threading.Thread(target=audio_worker, daemon=True)
worker_thread.start()

print("Close the plot window to stop the stream...")
plt.show()

stop_event.set()
worker_thread.join(timeout=1)
print("Stopping the stream...")
