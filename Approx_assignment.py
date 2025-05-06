import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

# Loading the guitar riff audio
filename = "guitar_riff.mp3"
y, sr = librosa.load(filename, sr=None)
print(f"Loaded {filename} | Duration: {len(y)/sr:.2f}s | Sample Rate: {sr}Hz")

# Apply FFT
Y = np.fft.fft(y)
freqs = np.fft.fftfreq(len(y), d=1 / sr)

# Plotting the frequency spectrum before filtering
plt.figure(figsize=(12, 4))
plt.xlim(0, 3000)
plt.plot(freqs[: len(freqs) // 2], np.abs(Y[: len(Y) // 2]))
plt.title("Original Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.tight_layout()
plt.show()

# Applying a simple Low-Pass Filter
cutoff = 1000
Y_filtered = Y.copy()
Y_filtered[np.abs(freqs) > cutoff] = 0

# Reconstructing the signal using inverse FFT
y_filtered = np.fft.ifft(Y_filtered).real

# Plotting time domain waveforms (original vs. filtered)
time = np.linspace(0, len(y) / sr, num=len(y))

plt.figure(figsize=(14, 5))

# Optional zoom (uncomment below to zoom into first second for clarity)
time = time[:sr]
y = y[:sr]
y_filtered = y_filtered[:sr]

plt.plot(time, y, label="Original", linewidth=1.2, alpha=0.7, color="blue")
plt.plot(
    time,
    y_filtered,
    label="Filtered (Low-pass)",
    linewidth=1.5,
    alpha=0.9,
    color="lightcoral",
)

plt.title("Time Domain: Original vs. Filtered Signal", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(loc="upper right", fontsize=12, frameon=True, facecolor="white")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# Plot the frequency spectrum after filtering
plt.figure(figsize=(12, 4))
plt.xlim(0, 1200)
plt.plot(
    freqs[: len(freqs) // 2],
    np.abs(Y_filtered[: len(Y_filtered) // 2]),
    color="crimson",
)
plt.title("Filtered Frequency Spectrum (Low-pass)", fontsize=14)
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Magnitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Saving the filtered audio
sf.write("guitar_riff_filtered.wav", y_filtered, sr)
print("Filtered audio saved as 'guitar_riff_filtered.wav'")
