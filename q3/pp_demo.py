import soundfile as sf
import torch
from privacymodule import privacy_transform


file_path = "../data/librispeech/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

signal, sr = sf.read(file_path)
signal = torch.tensor(signal, dtype=torch.float32)

if signal.ndim > 1:
    signal = signal[:, 0]

signal = signal.unsqueeze(0)

transformed = privacy_transform(signal)

sf.write("original.wav", signal.squeeze().numpy(), sr)
sf.write("transformed.wav", transformed.squeeze().detach().numpy(), sr)

print("Saved original.wav and transformed.wav")