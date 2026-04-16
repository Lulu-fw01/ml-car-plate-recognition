import csv


def save_errors_to_file(errors: list[dict], filename: str):
    fieldnames = ["filename", "ground_truth", "prediction", "cer", "s", "d", "i"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if errors:
            writer.writerows(errors)
