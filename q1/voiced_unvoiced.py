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
# Framing
# -------------------------------
def framing(signal, sr, frame_size=0.025, frame_stride=0.01):
    frame_length = int(frame_size * sr)
    frame_step = int(frame_stride * sr)

    signal_length = len(signal)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

    pad_length = num_frames * frame_step + frame_length
    pad_signal = np.append(signal, np.zeros(pad_length - signal_length))

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )

    frames = pad_signal[indices.astype(np.int32)]
    return frames


# -------------------------------
# Cepstrum
# -------------------------------
def compute_cepstrum(frame, NFFT=512):
    spectrum = np.fft.fft(frame, NFFT)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real
    return cepstrum


# -------------------------------
# Voiced/Unvoiced Detection
# -------------------------------
def is_voiced(cepstrum, sr, threshold=0.1):
    # pitch range ~ 50Hz–400Hz
    min_quef = int(sr / 400)
    max_quef = int(sr / 50)

    pitch_region = cepstrum[min_quef:max_quef]

    peak = np.max(pitch_region)

    return peak > threshold


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    file_path = "../data/librispeech/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

    sr, signal = load_audio(file_path)
    frames = framing(signal, sr)

    voiced_flags = []

    for frame in frames:
        cep = compute_cepstrum(frame)
        voiced = is_voiced(cep, sr)
        voiced_flags.append(1 if voiced else 0)

    voiced_flags = np.array(voiced_flags)

    print("Voiced frames:", np.sum(voiced_flags))
    print("Unvoiced frames:", len(voiced_flags) - np.sum(voiced_flags))

    # Visualization
    plt.figure(figsize=(12, 4))
    plt.plot(voiced_flags, label="Voiced(1)/Unvoiced(0)")
    plt.title("Voiced vs Unvoiced Detection")
    plt.xlabel("Frame Index")
    plt.ylabel("Voiced/Unvoiced")
    plt.legend()

    plt.savefig("voiced_unvoiced.png")
    plt.close()