import torch


def decode_greedy(alphabet: str, blank_idx: int, logits: torch.Tensor) -> list[str]:
    decoded_texts = []
    predictions = logits.argmax(dim=-1).permute(1, 0)

    for pred in predictions:
        chars = []
        prev = -1
        for idx in pred:
            idx = idx.item()
            if idx != prev and idx != blank_idx:
                if idx < len(alphabet):
                    chars.append(alphabet[idx])
            prev = idx
        decoded_texts.append("".join(chars))

    return decoded_texts
