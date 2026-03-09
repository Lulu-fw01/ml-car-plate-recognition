import torch


def ctc_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)

    alphabet = "0123456789ABEKMHOPCTYX"
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}

    targets = []
    target_lengths = []
    for label in labels:
        idxs = [char_to_idx[c] for c in label if c in char_to_idx]
        targets.extend(idxs)
        target_lengths.append(len(idxs))

    targets = torch.tensor(targets, dtype=torch.long)
    input_lengths = torch.full(
        size=(len(images),), fill_value=images.size(-1) // 4, dtype=torch.long
    )  # T = W/4 ≈ 32
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, input_lengths, target_lengths
