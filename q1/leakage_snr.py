import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


# -------------------------------
# Load Audio
# -------------------------------
def load_audio(file_path):
    signal, sr = sf.read(file_path)

    if signal.ndim > 1:
        signal = signal[:, 0]

    signal = signal / np.max(np.abs(signal))
    return sr, signal


# -------------------------------
# Get one frame (for analysis)
# -------------------------------
def get_frame(signal, sr, frame_size=0.025):
    frame_length = int(frame_size * sr)
    return signal[:frame_length]


# -------------------------------
# Apply window
# -------------------------------
def apply_window(frame, window_type):
    if window_type == "hamming":
        window = np.hamming(len(frame))
    elif window_type == "hanning":
        window = np.hanning(len(frame))
    else:
        window = np.ones(len(frame))  # rectangular

    return frame * window


# -------------------------------
# FFT Spectrum
# -------------------------------
def compute_fft(frame, NFFT=512):
    fft = np.fft.rfft(frame, NFFT)
    magnitude = np.abs(fft)
    return magnitude


# -------------------------------
# Spectral Leakage Measure
# -------------------------------
def spectral_leakage(spectrum):
    total_energy = np.sum(spectrum**2)
    peak_energy = np.max(spectrum**2)

    leakage = (total_energy - peak_energy) / total_energy
    return leakage


# -------------------------------
# SNR Calculation
# -------------------------------
def compute_snr(signal, noisy_signal):
    signal_power = np.mean(signal**2)
    noise_power = np.mean((signal - noisy_signal)**2)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    file_path = "../data/librispeech/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

    sr, signal = load_audio(file_path)
    frame = get_frame(signal, sr)

    windows = ["rectangular", "hamming", "hanning"]

    results = []

    plt.figure(figsize=(12, 6))

    for i, w in enumerate(windows):
        windowed = apply_window(frame, w)
        spectrum = compute_fft(windowed)

        leakage = spectral_leakage(spectrum)

        # simulate noise for SNR
        noise = np.random.normal(0, 0.01, len(frame))
        noisy_signal = windowed + noise
        snr = compute_snr(windowed, noisy_signal)

        results.append((w, leakage, snr))

        # Plot spectrum
        plt.subplot(1, 3, i + 1)
        plt.plot(spectrum)
        plt.title(f"{w}\nLeakage={leakage:.4f}, SNR={snr:.2f}dB")

    plt.tight_layout()
    plt.savefig("leakage_snr.png")
    plt.close()

    # Print results
    print("\nWindow Comparison:")
    for w, leak, snr in results:
        print(f"{w} -> Leakage: {leak:.4f}, SNR: {snr:.2f} dB")