from rapidfuzz.distance import Levenshtein


def calculate_prf(predictions: list[str], targets: list[str]) -> dict:
    tp, fp, fn = 0, 0, 0

    for pred, target in zip(predictions, targets):
        # Получаем правки: (substitutions, deletions, insertions)
        # s1=target, s2=pred (чтобы d было удалением из оригинала, а i - лишним в предсказании)
        edit = Levenshtein.editops(target, pred)
        s, d, i = edit.count("replace"), edit.count("delete"), edit.count("insert")

        # Математика для OCR:
        # Совпавшие символы (TP) = Длина оригинала - Удаления - Замены
        current_tp = len(target) - d - s

        tp += max(0, current_tp)
        fp += i + s  # Лишнее + неверное
        fn += d + s  # Пропущенное + неверное

    prec_denom = tp + fp
    rec_denom = tp + fn

    precision = tp / prec_denom if prec_denom > 0 else 0
    recall = tp / rec_denom if rec_denom > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
