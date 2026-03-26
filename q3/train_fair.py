import torch
import torch.nn as nn


def fairness_loss(outputs, labels):
    # simple group balancing proxy
    unique = torch.unique(labels)

    loss = 0
    for u in unique:
        mask = labels == u
        group_loss = torch.mean(outputs[mask])
        loss += group_loss

    return loss / len(unique)