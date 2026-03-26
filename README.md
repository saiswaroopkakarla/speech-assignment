#  Speech Processing & Representation Learning Assignment

##  Overview

This project implements a complete speech processing pipeline including:

* Manual MFCC feature extraction
* Spectral analysis (leakage & SNR)
* Voiced/Unvoiced segmentation using cepstrum
* Transformer-based phonetic alignment
* Disentangled speaker recognition (paper reproduction)
* Bias auditing, privacy-preserving transformations, and fairness learning

---

#  Project Structure

```
speech_assignment/
│
├── q1/
│   ├── mfcc_manual.py
│   ├── leakage_snr.py
│   ├── voiced_unvoiced.py
│   ├── phonetic_mapping.py
│   ├── q1_report.pdf
│
├── q2/
│   ├── train.py
│   ├── eval.py
│   ├── disentangled_model.pth
│   ├── model_comparison.png
│   ├── review.pdf
│   ├── q2_readme.md
│   ├── results/
│
├── q3/
│   ├── audit.py
│   ├── privacymodule.py
│   ├── pp_demo.py
│   ├── train_fair.py
│   ├── audit_distribution.png
│   ├── top_speakers.png
│   ├── original.wav
│   ├── transformed.wav
│   ├── q3_report.pdf
│
├── data/
├── requirements.txt
└── README.md
```

---

#  Installation

## 1. Create environment

```
python3 -m venv venv
source venv/bin/activate
```

## 2. Install dependencies

```
pip install -r requirements.txt
```

---

#  Dataset

We use:

* **LibriSpeech (train-clean-100)**

Download:

```
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xvzf train-clean-100.tar.gz
```

Place inside:

```
data/librispeech/
```

---

#  How to Run

---

## 🔹 Q1: Feature Extraction & Analysis

### MFCC Pipeline

```
python q1/mfcc_manual.py
```

### Spectral Leakage & SNR

```
python q1/leakage_snr.py
```

### Voiced/Unvoiced Detection

```
python q1/voiced_unvoiced.py
```

### Phonetic Mapping (Wav2Vec2)

```
python q1/phonetic_mapping.py
```

---

## 🔹 Q2: Disentangled Speaker Recognition

### Train model

```
python q2/train.py
```

### Evaluate model

```
python q2/eval.py
```

Outputs:

* Model accuracy comparison
* `model_comparison.png`

---

## 🔹 Q3: Bias, Privacy & Fairness

### Dataset audit

```
python q3/audit.py
```

### Privacy transformation demo

```
python q3/pp_demo.py
```

Outputs:

* `audit_distribution.png`
* `top_speakers.png`
* `original.wav`
* `transformed.wav`

---

#  Key Results

| Component          | Result |
| ------------------ | ------ |
| Baseline Accuracy  | ~0.5%  |
| Disentangled Model | ~30%   |
| Improvement        | ~50x   |

---

#  Key Concepts Implemented

* MFCC (from scratch)
* Cepstral analysis
* Spectral leakage analysis
* Transformer-based alignment
* Gradient Reversal Layer (GRL)
* Disentangled representation learning
* Bias auditing
* Privacy-preserving transformations
* Fairness-aware loss

---

#  Notes

* Training performed on CPU (GPU optional)
* Results may vary slightly depending on dataset subset
* Environment labels are synthetically generated (noise + reverb)

---

#  Conclusion

This project demonstrates:

* End-to-end speech processing pipeline
* Robust speaker recognition using disentanglement
* Ethical considerations including bias and privacy

---

#  Author
Name: Kakarla Sai Swaroop
Roll Number: M25DE1023
Mail: m25de1023@iitj.ac.in

---

```
```
