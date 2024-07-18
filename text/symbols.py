""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """
try:
    from text.dictionary import english, dutch, french, german, \
        indonesian, italian, japanese, korean, pinyin, \
        polish, portuguese, russian, spanish, vietnamese
except:
    from dictionary import english, dutch, french, german, \
        indonesian, italian, japanese, korean, pinyin, \
        polish, portuguese, russian, spanish, vietnamese

import pdb

_pad = ['@pad']
_unk = ['@unk']
_eos = ['@eos']
_sos = ['@sos']
_mask = ['@mask']

# _pad = ["_"]
# _mask_token = ["@MASK"]
_silences = ["@sp", "@spn", "@sil"]

#################### For all languages ####################
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
unique_symbol = True
if unique_symbol:
    _english = ["@" + s.replace("en_", "") for s in english.valid_symbols]
    _dutch = ["@" + s.replace("du_", "") for s in dutch.valid_symbols]
    _french = ["@" + s.replace("fr_", "") for s in french.valid_symbols]
    _german = ["@" + s.replace("ge_", "") for s in german.valid_symbols]
    _indonesian = ["@" + s.replace("in_", "") for s in indonesian.valid_symbols]
    _italian = ["@" + s.replace("it_", "") for s in italian.valid_symbols]
    _korean = ["@" + s.replace("ko_", "") for s in korean.valid_symbols]
    _pinyin = ["@" + s.replace("cn_", "") for s in pinyin.valid_symbols]
    _polish = ["@" + s.replace("po_", "") for s in polish.valid_symbols]
    _portuguese = ["@" + s.replace("por_", "") for s in portuguese.valid_symbols]
    _russian = ["@" + s.replace("ru_", "") for s in russian.valid_symbols]
    _spanish = ["@" + s.replace("sp_", "") for s in spanish.valid_symbols]
    _vietnamese = ["@" + s.replace("vn_", "") for s in vietnamese.valid_symbols]
    _japanese = ["@" + s.replace("jp_", "") for s in japanese.valid_symbols]
    _all_letters = _english + _dutch + _french + _german + _indonesian + _italian + \
        _korean + _pinyin + _polish + _portuguese + _russian + _spanish + _vietnamese + _japanese
    _all_letters = sorted(list(set(_all_letters)))
else:
    _english = ["@" + s.replace("en_", "") for s in english.valid_symbols]
    _dutch = ["@" + s.replace("du_", "") for s in dutch.valid_symbols]
    _french = ["@" + s for s in french.valid_symbols]
    _german = ["@" + s for s in german.valid_symbols]
    _indonesian = ["@" + s for s in indonesian.valid_symbols]
    _italian = ["@" + s for s in italian.valid_symbols]
    _korean = ["@" + s for s in korean.valid_symbols]
    _pinyin = ["@" + s for s in pinyin.valid_symbols]
    _polish = ["@" + s for s in polish.valid_symbols]
    _portuguese = ["@" + s for s in portuguese.valid_symbols]
    _russian = ["@" + s for s in russian.valid_symbols]
    _spanish = ["@" + s for s in spanish.valid_symbols]
    _vietnamese = ["@" + s for s in vietnamese.valid_symbols]
    _japanese = ["@" + s for s in japanese.valid_symbols]
    _all_letters = _english + _dutch + _french + _german + _indonesian + _italian + \
        _korean + _pinyin + _polish + _portuguese + _russian + _spanish + _vietnamese + _japanese

symbols = (
    _pad + _unk + _eos + _sos + _mask
    + _all_letters
    + _silences
    )