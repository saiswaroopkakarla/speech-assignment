import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# -------------------------------
# Load Audio
# -------------------------------
def load_audio(file_path):
    signal, sr = sf.read(file_path)

    if signal.ndim > 1:
        signal = signal[:, 0]

    return signal, sr


# -------------------------------
# Load Model
# -------------------------------
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model


# -------------------------------
# Get Predictions + Frame-wise logits
# -------------------------------
def get_logits(signal, sr, processor, model):
    inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    return logits


# -------------------------------
# Convert logits to timestamps
# -------------------------------
def get_predicted_timestamps(logits, sr, signal_length):
    pred_ids = torch.argmax(logits, dim=-1)

    # map frame index → time
    num_frames = logits.shape[1]
    duration = signal_length / sr

    time_per_frame = duration / num_frames

    timestamps = []

    for i, token in enumerate(pred_ids[0]):
        if token != 0:  # ignore blank
            timestamps.append(i * time_per_frame)

    return np.array(timestamps)


# -------------------------------
# Dummy manual segmentation (from your previous step)
# -------------------------------
def get_manual_boundaries(num_frames, sr, signal_length):
    duration = signal_length / sr
    frame_times = np.linspace(0, duration, num_frames)
    return frame_times


# -------------------------------
# RMSE Calculation
# -------------------------------
def compute_rmse(manual, predicted):
    min_len = min(len(manual), len(predicted))
    manual = manual[:min_len]
    predicted = predicted[:min_len]

    rmse = np.sqrt(np.mean((manual - predicted) ** 2))
    return rmse


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    file_path = "../data/librispeech/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

    signal, sr = load_audio(file_path)

    processor, model = load_model()

    logits = get_logits(signal, sr, processor, model)

    predicted_timestamps = get_predicted_timestamps(logits, sr, len(signal))

    # simulate manual segmentation (replace later if needed)
    manual_boundaries = get_manual_boundaries(len(predicted_timestamps), sr, len(signal))

    rmse = compute_rmse(manual_boundaries, predicted_timestamps)

    print("Predicted timestamps:", predicted_timestamps[:10])
    print("RMSE:", rmse)