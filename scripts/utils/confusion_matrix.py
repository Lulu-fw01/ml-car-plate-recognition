from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv


def update_confusion_matrix(
    current_cm: np.ndarray, pred: str, target: str, labels: list
):
    y_true = []
    y_pred = []

    max_len = max(len(pred), len(target))
    for i in range(max_len):
        p = pred[i] if i < len(pred) else "<DEL>"
        t = target[i] if i < len(target) else "<INS>"
        y_true.append(t)
        y_pred.append(p)

    new_cm = confusion_matrix(y_true, y_pred, labels=labels)

    return current_cm + new_cm


def save_confusion_matrix(cm: dict, filename: str):
    all_chars = set()
    for target_dict in cm.values():
        all_chars.update(target_dict.keys())
    all_chars.add("<DEL>")
    all_chars.add("<INS>")
    all_chars = sorted(
        all_chars, key=lambda x: (x == "<DEL>", x == "<INS>", x == "<PAD>", x)
    )

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Target\\Pred"] + all_chars)

        for target_char in all_chars:
            row = [target_char]
            for pred_char in all_chars:
                row.append(cm.get(target_char, {}).get(pred_char, 0))
            writer.writerow(row)


def save_to_csv(cm_array, labels, filename="matrix.csv"):
    df = pd.DataFrame(cm_array, index=labels, columns=labels)
    df.to_csv(filename, encoding="utf-8")


def save_to_png(cm_array, labels, filename="matrix.png"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.title("OCR Confusion Matrix")
    plt.savefig(filename)
    plt.close()
