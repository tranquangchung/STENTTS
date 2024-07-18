import sys
sys.path.append("..")
import numpy as np
import re
import argparse
from string import punctuation
import os
import json
import yaml
import numpy as np
from g2p_en import G2p
from pypinyin import pinyin, Style
from text import text_to_sequence
import pyopenjtalk
import subprocess
from utils.tools import add_prefix2phone, pp_symbols, openjtalk2julius
from korean_romanizer.romanizer import Romanizer
import pdb

abbv_language = {
    "dutch": "nl",
    "french": "fr-fr",
    "german": "de",
    "indonesian": "id",
    "italian": "it",
    "korean": "ko",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "vietnamese": "vi",
    "english": "en-us",
}
path_absolute = os.path.dirname(os.path.abspath(__file__))

# def openjtalk2julius(p3):
#     if p3 in ['A','I','U',"E", "O"]:
#         return "jp_" + p3.lower()
#     if p3 == 'cl':
#         return 'q'
#     if p3 == 'pau':
#         return 'sp'
#     if p3 == 'sil':
#         return 'sil'
#     return "jp_" + p3.lower()

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def load_lexicon_phone_mfa(filename):
    lexicon_phone = dict()
    with open(filename, encoding='utf-8') as f:
        for index, line in enumerate(f):
            lexicon, phone = line.strip().split("\t")
            lexicon_phone[lexicon] = phone.strip()
    return lexicon_phone

def preprocess_english(text, preprocess_config, **kwargs):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(os.path.join(path_absolute,"..", "lexicon/dictionary/english_espeak.txt"))

    phones = []
    words = text.split(" ")
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        elif w in [",", "."]:
            phones += ["sp"]
        else:
            command = "echo {0} | phonemize -l {1} -b espeak --language-switch remove-flags -p '-' -s '|' --strip".format(
                w, abbv_language["english"])
            phoneme = subprocess.getstatusoutput(command)[1].split("\n")[-1]
            phoneme = phoneme.replace("-", " ")
            phoneme = phoneme.strip().split()
            if len(phoneme) == 0: continue
            phones += phoneme
            print("Out of Vocabulary: ", w, phoneme)
    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config, **kwargs):
    lexicon = read_lexicon(os.path.join(path_absolute,".." ,"lexicon/dictionary/pinyin-lexicon-r.txt"))

    phones = []
    if kwargs.get("use_pinyin", None): # use pinyin
        pinyins = text.strip().split() if type(text) == str else text
    else: # None
        pinyins = [
            p[0]
            for p in pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
        ]

    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")
    use_prefix = kwargs.get("use_prefix", False)
    if use_prefix:
        phones = add_prefix2phone(phones, lang="chinese")
    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_japanese(text, preprocess_config, **kwargs):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    phones, accents = pp_symbols(fullcontext_labels)
    # phones = [openjtalk2julius(p) for p in phones if p != '']
    phones = [openjtalk2julius(p, prefix="") for p in phones if p != '']
    use_prefix = kwargs.get("use_prefix", False)
    if use_prefix:
        phones = add_prefix2phone(phones, lang="japanese")

    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_korean(text, preprocess_config, **kwargs):
    language = "korean"
    lexicon = load_lexicon_phone_mfa(os.path.join(path_absolute,"..", f"lexicon/dictionary/korean_espeak.txt"))
    if kwargs.get("hangeul", None): # use pinyin
        r = Romanizer(text)
        text = r.romanize()
    else:
        text = text.strip()
    words = text.split(" ")
    phones = []
    for word in words:
        if word in lexicon:
            phones += lexicon[word.lower()].split()
        elif word in [",", "."]:
            phones += "sp"
        else:
            command = "echo {0} | phonemize -l {1} -b espeak --language-switch remove-flags -p '-' -s '|' --strip".format(
                word, abbv_language[language])
            phoneme = subprocess.getstatusoutput(command)[1].split("\n")[-1]
            phoneme = phoneme.replace("-", " ")
            phoneme = phoneme.strip().split()
            if len(phoneme) == 0: continue
            phones += phoneme
            print("Out of Vocabulary: ", word, phoneme)
    use_prefix = kwargs.get("use_prefix", False)
    if use_prefix:
        phones = add_prefix2phone(phones, lang=language)
    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)

def preprocess_language(text, preprocess_config, language, **kwargs):
    lexicon = load_lexicon_phone_mfa(os.path.join(path_absolute,"..", f"lexicon/dictionary/{language}_espeak.txt"))
    words = text.split(" ")
    phones = []
    for word in words:
        if word in lexicon:
            phones += lexicon[word.lower()].split()
        elif word in [",","."]:
            phones += "sp"
        else:
            command = "echo {0} | phonemize -l {1} -b espeak --language-switch remove-flags -p '-' -s '|' --strip".format(word, abbv_language[language])
            phoneme = subprocess.getstatusoutput(command)[1].split("\n")[-1]
            phoneme = phoneme.replace("-", " ")
            phoneme = phoneme.strip().split()
            if len(phoneme) == 0: continue
            phones += phoneme
            print("Out of Vocabulary: ",word, phoneme)
    use_prefix = kwargs.get("use_prefix", False)
    if use_prefix:
        phones = add_prefix2phone(phones, lang=language)
    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def processes_all_languages(text, language, preprocess_config, **kwargs):
    if language == "chinese":
        return preprocess_mandarin(text, preprocess_config, **kwargs)
    elif language == "english":
        return preprocess_english(text, preprocess_config, **kwargs)
    elif language == "japanese":
        return preprocess_japanese(text, preprocess_config, **kwargs)
    elif language == "korean":
        return preprocess_korean(text, preprocess_config, **kwargs)
    else:
        return preprocess_language(text, preprocess_config, language, **kwargs)