from fireredtts.modules.text_normalizer.regex_common import *
from sentencex import segment
import re


symbol_reduction = {
    "「": '"',
    "」": '"',
    "｀": '"',
    "〝": '"',
    "〞": '"',
    "‟": '"',
    "„": '"',
    "｛": "(",
    "｝": ")",
    "【": "(",
    "】": ")",
    "〖": "(",
    "〗": ")",
    "〔": "(",
    "〕": ")",
    "〘": "(",
    "〙": ")",
    "《": "(",
    "》": ")",
    "｟": "(",
    "｠": ")",
    "〚": "(",
    "〛": ")",
    "『": '"',
    "』": '"',
    "｢": '"',
    "｣": '"',
    "{": "(",
    "}": ")",
    "〈": "(",
    "〉": ")",
    "•": "·",
    "‧": "·",
    "〰": "…",
    "﹏": "…",
    "〜": "~",
    "～": "~",
    "＋": "+",
    "､": "、",
    "｡": "。",
    "︐": "，",
    "﹐": "，",
    "︑": "、",
    "﹑": "、",
    "︒": "。",
    "︓": "：",
    "﹕": "：",
    "︔": "；",
    "﹔": "；",
    "︕": "！",
    "﹗": "！",
    "︖": "？",
    "﹖": "？",
    "﹙": "(",
    "﹚": ")",
    "﹪": "%",
    "﹠": "&",
    "＞": ">",
    "|": "、",
    "＝": "=",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "％": "%",
    "μ": "u",
}


strong_break = re.compile("([。”;；!！：…?？）\)\]』】」}~\r\n]| \.)", re.UNICODE)
weak_break = re.compile(
    "["
    "\U00002702-\U000027B0\U0001f926-\U0001f937\U00010000-\U0001fbff\U00030000-\U0010ffff"
    "\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\ufe0f\u3030"
    "\t，,. ]",
    re.UNICODE,
)


def contains_chinese(text):
    return bool(chinese_regex.search(text))


def strip_kaomoji(text):
    return kaomoji_regex.sub(" ", text)


def is_chinese(char):
    return chinese_char_regex.match(char)


def is_eng_and_digit(char):
    return eng_and_digit_char_regex.match(char)


def is_upper_eng_and_digit(text):
    return upper_eng_and_digit_regex.match(text)


def is_valid_char(char):
    return valid_char_regex.match(char)


def is_digit(text):
    return digit_regex.match(text)


def f2b(ustr, exemption="。，："):
    half = []
    for u in ustr:
        num = ord(u)
        if num == 0x3000:
            half.append(" ")
        elif u in exemption:  # exemption
            half.append(u)
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xFEE0
            half.append(chr(num))
        else:
            half.append(u)
    return "".join(half)


def zh_text_split(text, length=80):
    if length == 0:
        return []
    if length == 1:
        return [c for c in length]
    if len(text) <= length:
        return [text]

    match_strong = re.search(strong_break, text[:length][::-1])
    match_weak = re.search(weak_break, text[:length][::-1])
    end_ind_strong = length - match_strong.start() if match_strong else 0
    end_ind_weak = length - match_weak.start() if match_weak else 0

    if end_ind_strong < length // 3:
        if end_ind_weak < length // 3:
            valid_max = max(end_ind_strong, end_ind_weak)
            if valid_max >= 3:
                return [text[:valid_max]] + zh_text_split(text[valid_max:])
            else:
                return [text[:length]] + zh_text_split(text[length:])
        else:
            return [text[:end_ind_weak]] + zh_text_split(text[end_ind_weak:])
    else:
        return [text[:end_ind_strong]] + zh_text_split(text[end_ind_strong:])


def text_split(text):
    if contains_chinese(text):
        substrings = list(segment("zh", text))
        new_substrings = []
        for s in substrings:
            if len(s) > 30:
                new_substrings += zh_text_split(s, length=30)
            else:
                new_substrings.append(s)
        substrings = new_substrings
    else:
        substrings = list(segment("en", text))

    # merge substrings
    final_substrings = []
    temp_string = ""
    for i in range(len(substrings)):
        s = substrings[i]
        s = s.replace("。", ",")

        if i == 0:
            temp_string = s
        else:
            temp_string += s

        if len(temp_string) > 30:
            final_substrings.append(temp_string)
            temp_string = ""

        if i == len(substrings) - 1:
            if temp_string != "":
                final_substrings.append(temp_string)
            break

    if len(final_substrings) >= 2:
        if len(final_substrings[-1]) < 15:
            final_substrings[-2] += final_substrings[-1]
        final_substrings = final_substrings[:-1]

    return final_substrings
