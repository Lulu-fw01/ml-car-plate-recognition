import jiwer


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
    res = jiwer.cer(
        reference=[pred],
        hypothesis=[target],
        truth_transform=jiwer.Compose([]),
        hypothesis_transform=jiwer.Compose([]),
        return_dict=True,
    )
    cer = res["cer"]
    s = (res["substitutions"],)
    d = res["deletions"]
    i = res["insertions"]
    return cer, s, d, i
