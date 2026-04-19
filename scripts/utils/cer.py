import jiwer
from rapidfuzz.distance import Levenshtein


def get_levenshtein_operations(s1: str, s2: str) -> tuple:
    res = jiwer.cer(
        reference=[s1],
        hypothesis=[s2],
        truth_transform=jiwer.Compose([]),
        hypothesis_transform=jiwer.Compose([]),
        return_dict=True,
    )
    distance = res["substitutions"] + res["deletions"] + res["insertions"]
    return distance, res["substitutions"], res["deletions"], res["insertions"]


def get_cer(pred: str, target: str) -> tuple:
    if not target:
        return (1.0 if pred else 0.0), 0, 0, len(pred)

    ops = Levenshtein.editops(target, pred)

    s, d, i = 0, 0, 0
    for op in ops:
        if op.tag == "replace":
            s += 1
        elif op.tag == "delete":
            d += 1
        elif op.tag == "insert":
            i += 1

    row_cer = (s + d + i) / len(target)
    return row_cer, s, d, i
