import re
import string

SYMBOLS_MAPPING = {
    "\n": "",
    "…": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "・": "",
    "·": "",
    "「": "'",
    "」": "'",
    "《": "'",
    "》": "'",
    "—": "",
    "～": "",
    "~": "",
    "：": ",",
    "；": ",",
    ";": ",",
    ":": ",",
    '"': "",
    "！": "。",
    "!": ".",
    "————": "，",
    "——": "，",
    "—": "，",
    "……": "，",
}

REPLACE_SYMBOL_REGEX = re.compile(
    "|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys())
)


EMOJI_REGEX = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)


def clean_text(text):
    # Clean the text
    text = text.strip()
    text = text.replace("\xa0", "")

    # Replace all chinese symbols with their english counterparts
    text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)

    # Remove emojis
    text = EMOJI_REGEX.sub(r"", text)

    # Remove continuous periods (...) and commas (,,,)
    text = re.sub(r"[.,]{2,}", lambda m: m.group()[0], text)

    return text


def utf_8_len(text):
    return len(text.encode("utf-8"))


def break_text(texts, length, splits: set):
    for text in texts:
        if utf_8_len(text) <= length:
            yield text
            continue

        curr = ""
        for char in text:
            curr += char

            if char in splits:
                yield curr
                curr = ""

        if curr:
            yield curr


def break_text_by_length(texts, length):
    for text in texts:
        if utf_8_len(text) <= length:
            yield text
            continue

        curr = ""
        for char in text:
            curr += char

            if utf_8_len(curr) >= length:
                yield curr
                curr = ""

        if curr:
            yield curr


def add_cleaned(curr, segments):
    curr = curr.strip()
    if curr and not all(c.isspace() or c in string.punctuation for c in curr):
        segments.append(curr)


def protect_float(text):
    # Turns 3.14 into <3_f_14> to prevent splitting
    return re.sub(r"(\d+)\.(\d+)", r"<\1_f_\2>", text)


def unprotect_float(text):
    # Turns <3_f_14> into 3.14
    return re.sub(r"<(\d+)_f_(\d+)>", r"\1.\2", text)


def split_text(text, length):
    text = clean_text(text)

    # Break the text into pieces with following rules:
    # 1. Split the text at ".", "!", "?" if text is NOT a float
    # 2. If the text is longer than length, split at ","
    # 3. If the text is still longer than length, split at " "
    # 4. If the text is still longer than length, split at any character to length

    texts = [text]
    texts = map(protect_float, texts)
    texts = break_text(texts, length, {".", "!", "?", "。", "！", "？"})
    texts = map(unprotect_float, texts)
    texts = break_text(texts, length, {",", "，"})
    texts = break_text(texts, length, {" "})
    texts = list(break_text_by_length(texts, length))

    # Then, merge the texts into segments with length <= length
    segments = []
    curr = ""

    for text in texts:
        if utf_8_len(curr) + utf_8_len(text) <= length:
            curr += text
        else:
            add_cleaned(curr, segments)
            curr = text

    if curr:
        add_cleaned(curr, segments)

    return segments
