import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import soundfile as sf
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# Gradient Reversal Layer (GRL)
# -------------------------------
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


# -------------------------------
# Encoder (1D CNN)
# -------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=-1)
        return x


# -------------------------------
# Full Model
# -------------------------------
class DisentangledModel(nn.Module):
    def __init__(self, num_speakers, num_envs):
        super().__init__()

        self.encoder = Encoder()
        self.speaker_head = nn.Linear(64, num_speakers)
        self.env_head = nn.Linear(64, num_envs)

    def forward(self, x, lambda_=1.0):
        features = self.encoder(x)

        speaker_logits = self.speaker_head(features)

        reversed_features = grad_reverse(features, lambda_)
        env_logits = self.env_head(reversed_features)

        return speaker_logits, env_logits


# -------------------------------
# Add Noise
# -------------------------------
def add_noise(signal, noise_level=0.02):
    noise = torch.randn_like(signal) * noise_level
    return signal + noise


# -------------------------------
# Add Reverb (FIXED)
# -------------------------------
def add_reverb(signal):
    # signal: (1, time)
    kernel = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32).view(1, 1, -1)

    signal = signal.unsqueeze(0)  # (1, 1, time)
    reverb = torch.nn.functional.conv1d(signal, kernel, padding=2)

    return reverb.squeeze(0)  # back to (1, time)


# -------------------------------
# Dataset
# -------------------------------
class SpeechDataset(Dataset):
    def __init__(self, root_dir, max_files=200):
        self.files = []
        self.labels = []

        speaker_map = {}
        speaker_id = 0

        for root, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(".flac"):
                    path = os.path.join(root, f)

                    spk = root.split("/")[-2]

                    if spk not in speaker_map:
                        speaker_map[spk] = speaker_id
                        speaker_id += 1

                    self.files.append(path)
                    self.labels.append(speaker_map[spk])

                    if len(self.files) >= max_files:
                        break

        self.num_speakers = len(set(self.labels))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        speaker = self.labels[idx]

        # ---- LOAD AUDIO ----
        signal, sr = sf.read(path)
        signal = torch.tensor(signal, dtype=torch.float32)

        if signal.ndim > 1:
            signal = signal[:, 0]

        signal = signal.unsqueeze(0)  # (1, time)

        # ---- ENVIRONMENT SIMULATION ----
        env = random.randint(0, 2)

        if env == 1:
            signal = add_noise(signal)
        elif env == 2:
            signal = add_reverb(signal)

        # ---- FINAL LENGTH FIX (CRITICAL) ----
        signal = signal[:, :16000]

        if signal.shape[1] < 16000:
            pad = 16000 - signal.shape[1]
            signal = F.pad(signal, (0, pad))

        return signal, speaker, env


# -------------------------------
# Training Loop
# -------------------------------
# def train():
#     dataset = SpeechDataset("../data/librispeech/LibriSpeech/train-clean-100", max_files=200)
#     dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#     model = DisentangledModel(num_speakers=dataset.num_speakers, num_envs=3)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(2):
#         for signals, speakers, envs in dataloader:
#             outputs_spk, outputs_env = model(signals)

#             loss_spk = criterion(outputs_spk, speakers)
#             loss_env = criterion(outputs_env, envs)

#             loss = loss_spk + 0.5 * loss_env

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch} Loss: {loss.item()}")

def train():
    dataset = SpeechDataset("../data/librispeech/LibriSpeech/train-clean-100", max_files=500)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DisentangledModel(num_speakers=dataset.num_speakers, num_envs=3)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 70  # 🔥 increase training

    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0

        for signals, speakers, envs in dataloader:
            outputs_spk, outputs_env = model(signals)

            loss_spk = criterion(outputs_spk, speakers)
            loss_env = criterion(outputs_env, envs)

            loss = loss_spk + 0.5 * loss_env

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    # 🔥 SAVE MODEL (IMPORTANT)
    torch.save(model.state_dict(), "disentangled_model.pth")
    print("Model saved as disentangled_model.pth")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    train()