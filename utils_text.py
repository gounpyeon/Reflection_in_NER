# utils_text.py
import re

_PUNCT = r"\.\,\:\;\?\!\％\%\·"
_QUOTES = r"\"\'\u2018\u2019\u201C\u201D"
_PARENS = r"\(\)\[\]\{\}"

def detok(tokens):
    s = " ".join(tokens)
    # 구두점 앞 공백 제거
    s = re.sub(rf"\s+([{_PUNCT}])", r"\1", s)
    # 여는 괄호/따옴표 뒤 공백 제거
    s = re.sub(rf"([{_QUOTES}{_PARENS}])\s+", r"\1", s)
    # 닫는 괄호/따옴표 앞 공백 제거
    s = re.sub(rf"\s+([{_PARENS}{_QUOTES}])", r"\1", s)
    # 중복 공백 정리
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def normalize_span_text(t):
    t = t.strip()
    t = t.strip('"\''"“”‘’")
    t = re.sub(r"\s+", " ", t)
    return t.lower()