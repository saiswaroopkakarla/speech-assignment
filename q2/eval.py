# # import torch
# # import torch.nn as nn
# # import os
# # import soundfile as sf
# # import random
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader


# # # -------------------------------
# # # Dataset (same as train)
# # # -------------------------------
# # class SpeechDataset(Dataset):
# #     def __init__(self, root_dir, max_files=100):
# #         self.files = []
# #         self.labels = []

# #         speaker_map = {}
# #         speaker_id = 0

# #         for root, _, filenames in os.walk(root_dir):
# #             for f in filenames:
# #                 if f.endswith(".flac"):
# #                     path = os.path.join(root, f)

# #                     spk = root.split("/")[-2]

# #                     if spk not in speaker_map:
# #                         speaker_map[spk] = speaker_id
# #                         speaker_id += 1

# #                     self.files.append(path)
# #                     self.labels.append(speaker_map[spk])

# #                     if len(self.files) >= max_files:
# #                         break

# #         self.num_speakers = len(set(self.labels))

# #     def __len__(self):
# #         return len(self.files)

# #     def __getitem__(self, idx):
# #         path = self.files[idx]
# #         speaker = self.labels[idx]

# #         signal, sr = sf.read(path)
# #         signal = torch.tensor(signal, dtype=torch.float32)

# #         if signal.ndim > 1:
# #             signal = signal[:, 0]

# #         signal = signal.unsqueeze(0)

# #         # FIX LENGTH
# #         signal = signal[:, :16000]
# #         if signal.shape[1] < 16000:
# #             pad = 16000 - signal.shape[1]
# #             signal = F.pad(signal, (0, pad))

# #         return signal, speaker


# # # -------------------------------
# # # Simple Encoder
# # # -------------------------------
# # class Encoder(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.conv = nn.Sequential(
# #             nn.Conv1d(1, 16, 5, 2),
# #             nn.ReLU(),
# #             nn.Conv1d(16, 32, 5, 2),
# #             nn.ReLU(),
# #             nn.Conv1d(32, 64, 5, 2),
# #             nn.ReLU(),
# #         )

# #     def forward(self, x):
# #         x = self.conv(x)
# #         return torch.mean(x, dim=-1)


# # # -------------------------------
# # # Baseline Model (NO GRL)
# # # -------------------------------
# # class BaselineModel(nn.Module):
# #     def __init__(self, num_speakers):
# #         super().__init__()
# #         self.encoder = Encoder()
# #         self.fc = nn.Linear(64, num_speakers)

# #     def forward(self, x):
# #         features = self.encoder(x)
# #         return self.fc(features)


# # # -------------------------------
# # # Evaluation Function
# # # -------------------------------
# # def evaluate(model, dataloader):
# #     model.eval()
# #     correct = 0
# #     total = 0

# #     with torch.no_grad():
# #         for signals, speakers in dataloader:
# #             outputs = model(signals)
# #             preds = torch.argmax(outputs, dim=1)

# #             correct += (preds == speakers).sum().item()
# #             total += speakers.size(0)

# #     return correct / total


# # # -------------------------------
# # # MAIN
# # # -------------------------------
# # if __name__ == "__main__":

# #     dataset = SpeechDataset("../data/librispeech/LibriSpeech/train-clean-100", max_files=100)
# #     dataloader = DataLoader(dataset, batch_size=8)

# #     model = BaselineModel(num_speakers=dataset.num_speakers)

# #     acc = evaluate(model, dataloader)



# #     print(f"Baseline Accuracy: {acc:.4f}")




# import torch
# import torch.nn as nn
# import os
# import soundfile as sf
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader


# # -------------------------------
# # Dataset
# # -------------------------------
# class SpeechDataset(Dataset):
#     def __init__(self, root_dir, max_files=100):
#         self.files = []
#         self.labels = []

#         speaker_map = {}
#         speaker_id = 0

#         for root, _, filenames in os.walk(root_dir):
#             for f in filenames:
#                 if f.endswith(".flac"):
#                     path = os.path.join(root, f)

#                     spk = root.split("/")[-2]

#                     if spk not in speaker_map:
#                         speaker_map[spk] = speaker_id
#                         speaker_id += 1

#                     self.files.append(path)
#                     self.labels.append(speaker_map[spk])

#                     if len(self.files) >= max_files:
#                         break

#         self.num_speakers = len(set(self.labels))

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         speaker = self.labels[idx]

#         signal, sr = sf.read(path)
#         signal = torch.tensor(signal, dtype=torch.float32)

#         if signal.ndim > 1:
#             signal = signal[:, 0]

#         signal = signal.unsqueeze(0)

#         signal = signal[:, :16000]
#         if signal.shape[1] < 16000:
#             pad = 16000 - signal.shape[1]
#             signal = F.pad(signal, (0, pad))

#         return signal, speaker


# # -------------------------------
# # Encoder
# # -------------------------------
# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 16, 5, 2),
#             nn.ReLU(),
#             nn.Conv1d(16, 32, 5, 2),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, 5, 2),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return torch.mean(x, dim=-1)


# # -------------------------------
# # Baseline Model
# # -------------------------------
# class BaselineModel(nn.Module):
#     def __init__(self, num_speakers):
#         super().__init__()
#         self.encoder = Encoder()
#         self.fc = nn.Linear(64, num_speakers)

#     def forward(self, x):
#         return self.fc(self.encoder(x))


# # -------------------------------
# # SAME MODEL AS TRAIN (IMPORTANT)
# # -------------------------------
# class DisentangledModel(nn.Module):
#     def __init__(self, num_speakers, num_envs):
#         super().__init__()
#         self.encoder = Encoder()
#         self.speaker_head = nn.Linear(64, num_speakers)
#         self.env_head = nn.Linear(64, num_envs)

#     def forward(self, x):
#         features = self.encoder(x)
#         speaker_logits = self.speaker_head(features)
#         return speaker_logits


# # -------------------------------
# # Evaluation
# # -------------------------------
# def evaluate(model, dataloader):
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for signals, speakers in dataloader:
#             outputs = model(signals)
#             preds = torch.argmax(outputs, dim=1)

#             correct += (preds == speakers).sum().item()
#             total += speakers.size(0)

#     return correct / total


# # -------------------------------
# # MAIN
# # -------------------------------
# if __name__ == "__main__":

#     dataset = SpeechDataset("../data/librispeech/LibriSpeech/train-clean-100", max_files=100)
#     dataloader = DataLoader(dataset, batch_size=8)

#     # Baseline
#     baseline = BaselineModel(num_speakers=dataset.num_speakers)
#     baseline_acc = evaluate(baseline, dataloader)

#     # Trained model
#     model = DisentangledModel(num_speakers=dataset.num_speakers, num_envs=3)
#     model.load_state_dict(torch.load("disentangled_model.pth"))

#     trained_acc = evaluate(model, dataloader)

#     print(f"Baseline Accuracy: {baseline_acc:.4f}")
#     print(f"Trained Model Accuracy: {trained_acc:.4f}")



import torch
import torch.nn as nn
import os
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -------------------------------
# Dataset
# -------------------------------
class SpeechDataset(Dataset):
    def __init__(self, root_dir, max_files=100):
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

        signal, sr = sf.read(path)
        signal = torch.tensor(signal, dtype=torch.float32)

        if signal.ndim > 1:
            signal = signal[:, 0]

        signal = signal.unsqueeze(0)

        signal = signal[:, :16000]
        if signal.shape[1] < 16000:
            pad = 16000 - signal.shape[1]
            signal = F.pad(signal, (0, pad))

        return signal, speaker


# -------------------------------
# Encoder
# -------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.mean(x, dim=-1)


# -------------------------------
# Baseline Model
# -------------------------------
class BaselineModel(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.encoder = Encoder()
        self.fc = nn.Linear(64, num_speakers)

    def forward(self, x):
        return self.fc(self.encoder(x))


# -------------------------------
# Trained Model
# -------------------------------
class DisentangledModel(nn.Module):
    def __init__(self, num_speakers, num_envs):
        super().__init__()
        self.encoder = Encoder()
        self.speaker_head = nn.Linear(64, num_speakers)
        self.env_head = nn.Linear(64, num_envs)

    def forward(self, x):
        features = self.encoder(x)
        return self.speaker_head(features)


# -------------------------------
# Evaluation
# -------------------------------
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, speakers in dataloader:
            outputs = model(signals)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == speakers).sum().item()
            total += speakers.size(0)

    return correct / total


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    dataset = SpeechDataset("../data/librispeech/LibriSpeech/train-clean-100", max_files=100)
    dataloader = DataLoader(dataset, batch_size=8)

    # Baseline
    baseline = BaselineModel(num_speakers=dataset.num_speakers)
    baseline_acc = evaluate(baseline, dataloader)

    # Trained model
    model = DisentangledModel(num_speakers=dataset.num_speakers, num_envs=3)
    model.load_state_dict(torch.load("disentangled_model.pth"))

    trained_acc = evaluate(model, dataloader)

    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Trained Model Accuracy: {trained_acc:.4f}")

    # -------------------------------
    # PLOT RESULTS
    # -------------------------------
    models = ["Baseline", "Disentangled"]
    accuracies = [baseline_acc, trained_acc]

    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracies)

    plt.title("Model Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")

    # annotate values
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

    # save
    plt.savefig("model_comparison.png")
    print("Plot saved as model_comparison.png")

    # show (optional)
    plt.show()