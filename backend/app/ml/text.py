"""Indonesian text normalisation shared by training (ml/) and serving (backend).

Pickled scikit-learn pipelines reference `normalize` by import path, so this module must stay
importable as `app.ml.text` wherever a model is loaded.
"""

import re

# Common informal spellings in chat, reviews and complaints -> standard form.
SLANG = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "tdk": "tidak", "tak": "tidak", "g": "tidak", "kagak": "tidak", "ndak": "tidak",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "untk": "untuk", "u/": "untuk",
    "sm": "sama", "sma": "sama", "krn": "karena", "karna": "karena", "krna": "karena",
    "jg": "juga", "jga": "juga", "aja": "saja", "aj": "saja", "doang": "saja",
    "bgt": "banget", "bngt": "banget", "bener": "benar", "bnr": "benar",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "dah": "sudah", "blm": "belum", "blom": "belum",
    "lg": "lagi", "lgi": "lagi", "tp": "tapi", "tpi": "tapi", "trs": "terus", "trus": "terus",
    "dr": "dari", "dri": "dari", "kl": "kalau", "kalo": "kalau", "klo": "kalau", "kalau2": "kalau",
    "gmn": "bagaimana", "gimana": "bagaimana", "gmna": "bagaimana", "knp": "kenapa", "napa": "kenapa",
    "brp": "berapa", "brapa": "berapa", "apa2": "apa", "sy": "saya", "aku": "saya", "gw": "saya",
    "gue": "saya", "w": "saya", "km": "kamu", "lu": "kamu", "lo": "kamu", "kmu": "kamu",
    "bs": "bisa", "bsa": "bisa", "jd": "jadi", "jdi": "jadi", "sbg": "sebagai", "pd": "pada",
    "org": "orang", "orng": "orang", "bln": "bulan", "thn": "tahun", "th": "tahun", "tgl": "tanggal",
    "hr": "hari", "mgg": "minggu", "jml": "jumlah", "jmlh": "jumlah", "tot": "total",
    "rata2": "rata-rata", "ratarata": "rata-rata", "msh": "masih", "masi": "masih",
    "bkn": "bukan", "bukn": "bukan", "emg": "memang", "emang": "memang", "mmg": "memang",
    "sgt": "sangat", "bgs": "bagus", "mantep": "mantap", "mantab": "mantap", "jelek2": "jelek",
    "mksh": "terima kasih", "makasih": "terima kasih", "thx": "terima kasih", "tq": "terima kasih",
    "tlg": "tolong", "tolongin": "tolong", "dong": "", "deh": "", "sih": "", "nih": "", "kok": "",
    "ya": "", "yah": "", "yaa": "", "kak": "", "bang": "", "min": "", "gan": "", "sis": "",
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"[@#]\w+")
_REPEAT = re.compile(r"(\w)\1{2,}")  # "bagusss" -> "baguss"
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|u/")


def normalize(text: str) -> str:
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _REPEAT.sub(r"\1\1", text)
    words = []
    for token in _TOKEN.findall(text):
        replacement = SLANG.get(token, token)
        if replacement:
            words.append(replacement)
    return " ".join(words)


def normalize_batch(texts) -> list[str]:
    return [normalize(t) for t in texts]
