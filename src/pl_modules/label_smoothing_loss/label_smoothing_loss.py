import torch
import torch.nn as nn


def label_smoothing_loss(alphabet: str, blank_idx: int, log_probs, targets, device):
    baseline_loss = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)

    T, B, C = log_probs.shape
    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

    targets_list = []
    target_lengths_list = []
    for text in targets:
        t = [alphabet.index(c) for c in text if c in alphabet]
        targets_list.extend(t)
        target_lengths_list.append(len(t))

    targets = torch.tensor(targets_list, dtype=torch.long, device=device)
    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long, device=device)

    return baseline_loss(log_probs, targets, input_lengths, target_lengths)
