def calculate_per_position_accuracy(
    predictions: list[str], targets: list[str], max_length: int = 9
) -> list[dict]:
    correct_counts = [0] * max_length
    total_counts = [0] * max_length

    for pred, target in zip(predictions, targets):
        for i, target_char in enumerate(target[:max_length]):
            pred_char = pred[i] if i < len(pred) else "<PAD>"

            total_counts[i] += 1
            if pred_char == target_char:
                correct_counts[i] += 1

    return [
        {
            "position": i + 1,
            "accuracy": round(correct / total, 4),
            "correct": correct,
            "total": total,
        }
        for i, (correct, total) in enumerate(zip(correct_counts, total_counts))
        if total > 0
    ]
