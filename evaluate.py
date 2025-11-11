# eval.py
import re
from typing import List, Dict, Tuple

def _norm(s: str):
    s = s.strip()
    s = s.strip('"\''"“”‘’")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def _tupleize(ents: List[Dict], with_type=True):
    tups = []
    for e in ents:
        txt = _norm(e["text"])
        typ = e.get("type", "")
        tups.append((txt, typ if with_type else ""))
    return set(tups)

def prf1(gold: List[Dict], pred: List[Dict], with_type=True) -> Tuple[float,float,float]:
    G = _tupleize(gold, with_type)
    P = _tupleize(pred, with_type)
    tp = len(G & P)
    fp = len(P - G)
    fn = len(G - P)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return prec, rec, f1