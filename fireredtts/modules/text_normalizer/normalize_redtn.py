# -*- coding: UTF-8 -*-

import os
import re
import regex
import unicodedata
from builtins import str as unicode

import fireredtts.modules.text_normalizer.redtn as tn

def preprocess_text(sentence):
    # preprocessing
    sentence = bytes(sentence, "utf-8").decode("utf-8", "ignore")
    sentence = regex.sub("[\p{Cf}--[\u200d]]", "", sentence, flags=regex.V1)
    sentence = regex.sub("\p{Co}", "", sentence)
    sentence = sentence.replace("\u00a0", " ")
    sentence = sentence.replace("\ufffd", "")
    sentence = regex.sub("\p{Zl}", "\n", sentence)
    sentence = regex.sub("\p{Zp}", "\n", sentence)
    sentence = sentence.strip()
    return sentence


def postprocess_text(sentence):
    sentence = sentence.replace("-", " ")
    sentence = sentence.replace("：", ":")
    sentence = sentence.replace("？", "?")

    lines = sentence.split("|")
    lines = [(l[:-4], l[-4:]) for l in lines if len(l) > 4]  # extract text and lang
    output = []
    for text, lang in lines:
        # model limitations
        text = re.sub(r"[#《·》]+", " ", text)
        text = re.sub(r"[&*%$^()：_/]+([A-Za-z])", r" \1", text)
        text = re.sub(r"([\u4e00-\u9fd5])[ ]+", r"\1", text)
        text = re.sub(r"[ ]+([\u4e00-\u9fd5])", r"\1", text)
        if lang == "[zh]":
            text = re.sub(r"[…~！，&*·%$^\[\]()：；!_/:;]+", "，", text)
            text = re.sub(r"[，。 ]+$", "。", text)
            text = re.sub(r"^[，。 ]+", "", text)
            text = re.sub(r"[ ]+,", "，", text)
            text = re.sub(r"[，]+", "，", text)
            text = re.sub(r"[ ]+", " ", text)
            if len(text) > 0 and text[-1] not in "。?":
                text = text + "。"
        elif lang == "[en]":
            text = re.sub(r"[…~！,&*·%$^，、\[\]()：；!_/:;]+", ",", text)
            text = re.sub(r"[,. ]+$", ".", text)
            text = re.sub(r"^[,. ]+", "", text)
            text = re.sub(r"[ ]+,", ",", text)
            text = re.sub(r"[,]+", ",", text)
            text = re.sub(r"[ ]+", " ", text)
            if len(text) > 0 and text[-1] not in ".?":
                text = text + "."
        else:
            print(lang)
        # text = text.lower()
        text = text + "<sbreak>"
        output.append((text, lang))
    return output


class TextNormalizer:
    def __init__(self, data_dir):
        self.tn_engine = tn.RedTN(data_dir)

    def tn(self, text):
        text = preprocess_text(text)
        text = self.tn_engine.tn(text).strip()
        text_lang_pairs = postprocess_text(text)
        return text_lang_pairs


if __name__ == "__main__":
    tn = TextNormalizer()
    print(tn.tn("2013，我的UR梦？UR Really ur Ur really, happy 2013."))
    print(tn.tn("2021年银行要为持卡人提供容差服务。"))
    print(tn.tn("你仍然有义务偿还欠款及所有应付利息。"))
