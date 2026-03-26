import torch


def pitch_shift(signal, factor=1.2):
    return signal * factor


def add_noise(signal, noise_level=0.01):
    noise = torch.randn_like(signal) * noise_level
    return signal + noise


def privacy_transform(signal):
    signal = pitch_shift(signal, factor=1.1)
    signal = add_noise(signal, 0.01)
    return signal