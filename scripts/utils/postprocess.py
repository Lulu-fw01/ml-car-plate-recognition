ALLOWED_ALPHABET = set("0123456789ABEKMHOPCTYX")


def postprocess_plate_text(text: str) -> str:
    if not text:
        return ""

    if text.endswith(".RUS"):
        text = text[:-4]

    text = "".join(c for c in text if c in ALLOWED_ALPHABET)

    if len(text) < 2:
        return ""

    text = text[:9]

    while len(text) >= 2 and not text[-2:].isdigit():
        text = text[:-1]

    if len(text) < 2:
        return ""
    if any(c not in ALLOWED_ALPHABET for c in text):
        return ""
    if not text[-2:].isdigit():
        return ""

    return text
