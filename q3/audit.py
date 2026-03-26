import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def audit_dataset(root_dir, max_files=200):
    speaker_counts = {}

    count = 0

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".flac"):
                speaker = root.split("/")[-2]

                if speaker not in speaker_counts:
                    speaker_counts[speaker] = 0

                speaker_counts[speaker] += 1
                count += 1

                if count >= max_files:
                    break

    return speaker_counts


if __name__ == "__main__":
    data_path = "../data/librispeech/LibriSpeech/train-clean-100"

    speaker_counts = audit_dataset(data_path)

    print("Total speakers:", len(speaker_counts))

    values = list(speaker_counts.values())

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=20)
    plt.title("Speaker Distribution")
    plt.xlabel("Samples per Speaker")
    plt.ylabel("Frequency")

    plt.savefig("audit_distribution.png")
    plt.show()

    # top 10 speakers
top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]

names = [x[0] for x in top_speakers]
counts = [x[1] for x in top_speakers]

plt.figure(figsize=(8, 4))
plt.bar(names, counts)

plt.title("Top 10 Speakers Distribution")
plt.xlabel("Speaker ID")
plt.ylabel("Samples")

plt.savefig("top_speakers.png")
plt.close()

print("Saved top_speakers.png")