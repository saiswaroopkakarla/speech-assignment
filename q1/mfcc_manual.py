# import numpy as np
# import matplotlib.pyplot as plt
# import soundfile as sf


# # -------------------------------
# # 1. Load Audio
# # -------------------------------
# def load_audio(file_path):
#     signal, sr = sf.read(file_path)
    
#     # Normalize
#     if signal.ndim > 1:
#         signal = signal[:, 0]  # take first channel if stereo
    
#     signal = signal / np.max(np.abs(signal))
#     return sr, signal


# # -------------------------------
# # 2. Pre-emphasis
# # -------------------------------
# def pre_emphasis(signal, alpha=0.97):
#     emphasized = np.append(signal[0], signal[1:] - alpha * signal[:-1])
#     return emphasized


# # -------------------------------
# # 3. Framing
# # -------------------------------
# def framing(signal, sr, frame_size=0.025, frame_stride=0.01):
#     frame_length = int(frame_size * sr)
#     frame_step = int(frame_stride * sr)

#     signal_length = len(signal)
#     num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

#     pad_length = num_frames * frame_step + frame_length
#     pad_signal = np.append(signal, np.zeros(pad_length - signal_length))

#     indices = (
#         np.tile(np.arange(0, frame_length), (num_frames, 1)) +
#         np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
#     )

#     frames = pad_signal[indices.astype(np.int32)]
#     return frames


# # -------------------------------
# # 4. Windowing
# # -------------------------------
# def apply_window(frames, window_type="hamming"):
#     if window_type == "hamming":
#         window = np.hamming(frames.shape[1])
#     elif window_type == "hanning":
#         window = np.hanning(frames.shape[1])
#     else:
#         window = np.ones(frames.shape[1])  # rectangular
    
#     return frames * window


# # -------------------------------
# # MAIN (TEST PIPELINE)
# # -------------------------------
# if __name__ == "__main__":
    
#     # ⚠️ CHANGE THIS PATH if needed
#     file_path = "../data/librispeech/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

#     sr, signal = load_audio(file_path)
#     emphasized = pre_emphasis(signal)
#     frames = framing(emphasized, sr)
#     windowed = apply_window(frames)

#     print("Sample Rate:", sr)
#     print("Signal shape:", signal.shape)
#     print("Frames shape:", frames.shape)

#     # Plot original signal
#     plt.figure(figsize=(10, 4))
#     plt.plot(signal)
#     plt.title("Original Signal")
#     plt.savefig("original_signal.png")
#     plt.close()

#     # Plot one frame
#     plt.figure(figsize=(10, 4))
#     plt.plot(windowed[0])
#     plt.title("Windowed Frame (First Frame)")
#     plt.savefig("windowed_frame.png")
#     plt.close()



import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fftpack import dct
import matplotlib
matplotlib.use('Agg')


# -------------------------------
# 1. Load Audio
# -------------------------------
def load_audio(file_path):
    signal, sr = sf.read(file_path)

    # handle stereo
    if signal.ndim > 1:
        signal = signal[:, 0]

    # normalize
    signal = signal / np.max(np.abs(signal))
    return sr, signal


# -------------------------------
# 2. Pre-emphasis
# -------------------------------
def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


# -------------------------------
# 3. Framing
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
# 4. Windowing
# -------------------------------
def apply_window(frames, window_type="hamming"):
    if window_type == "hamming":
        window = np.hamming(frames.shape[1])
    elif window_type == "hanning":
        window = np.hanning(frames.shape[1])
    else:
        window = np.ones(frames.shape[1])  # rectangular

    return frames * window


# -------------------------------
# 5. FFT + Power Spectrum
# -------------------------------
def compute_fft(frames, NFFT=512):
    fft_frames = np.fft.rfft(frames, NFFT)
    power_spectrum = (1.0 / NFFT) * (np.abs(fft_frames) ** 2)
    return power_spectrum


# -------------------------------
# 6. Mel Filterbank
# -------------------------------
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)


def mel_filterbank(sr, NFFT=512, nfilt=40):
    low_mel = 0
    high_mel = hz_to_mel(sr / 2)

    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))

    for m in range(1, nfilt + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)

        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)

    return fbank


# -------------------------------
# 7. Apply Mel Filter + Log
# -------------------------------
def apply_mel_filter(power_spectrum, fbank):
    filter_banks = np.dot(power_spectrum, fbank.T)

    # avoid log(0)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)

    return np.log(filter_banks)


# -------------------------------
# 8. DCT → MFCC
# -------------------------------
def compute_mfcc(log_fbank, num_ceps=13):
    mfcc = dct(log_fbank, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    file_path = "../data/librispeech/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

    sr, signal = load_audio(file_path)
    emphasized = pre_emphasis(signal)
    frames = framing(emphasized, sr)
    windowed = apply_window(frames)

    power_spectrum = compute_fft(windowed)
    fbank = mel_filterbank(sr)
    log_fbank = apply_mel_filter(power_spectrum, fbank)
    mfcc = compute_mfcc(log_fbank)

    print("Sample Rate:", sr)
    print("Signal shape:", signal.shape)
    print("Frames shape:", frames.shape)
    print("MFCC shape:", mfcc.shape)

    # Plot MFCC heatmap
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc.T, aspect='auto', origin='lower')
    plt.title("MFCC")
    plt.xlabel("Frames")
    plt.ylabel("Coefficients")
    plt.colorbar()
    plt.savefig("mfcc.png")
    plt.close()