# adapters.py
import uuid
from typing import Dict, Any, List, Optional
from utils_text import detok
from utils_iob import iob_to_spans

_PICO_MAP = {"PAR": "P", "INT": "I", "COM": "C", "OUT": "O"}

def _decode_tags(tags: List, tag_names: Optional[List[str]]) -> List[str]:
    """
    tags가 정수라면 tag_names로 문자열로 매핑. 이미 문자열이면 그대로 반환.
    """
    if not tags:
        return []
    if isinstance(tags[0], int):
        if not tag_names:
            raise ValueError("Integer tags provided but no tag_names to decode them.")
        return [tag_names[t] for t in tags]
    return tags  # already strings

def _normalize_suffix_bio_fast(tags):
    out = []
    for t in tags:
        if t == "O":
            out.append("O"); continue
        if "_" in t:
            ent, bio = t.rsplit("_", 1)
            if bio in ("B","I","E","S"):
                out.append(f"{bio}-{ent}")
                continue
        out.append(t)
    return out

def adapt_conll(example: Dict[str, Any], split: str, tag_names: Optional[List[str]] = None):
    tokens = example["words"]
    tags = _decode_tags(example["ner"], tag_names)
    text = detok(tokens)
    spans = iob_to_spans(tokens, tags)
    return {
        "id": example.get("id") or str(uuid.uuid4()),
        "dataset": "conll",
        "split": split,
        "text": text,
        "schema": "generic",
        "labels": spans
    }

def adapt_wnut(example: Dict[str, Any], split: str, tag_names: Optional[List[str]] = None):
    tokens = example["tokens"]
    tags = _decode_tags(example["ner_tags"], tag_names)
    text = detok(tokens)
    spans = iob_to_spans(tokens, tags)
    return {
        "id": str(example.get("id") or uuid.uuid4()),
        "dataset": "wnut",
        "split": split,
        "text": text,
        "schema": "generic",
        "labels": spans
    }

def adapt_ebmnlp(example: Dict[str, Any], split: str, tag_names: Optional[List[str]] = None):
    tokens = example["tokens"]
    tags = _decode_tags(example["ner_tags"], tag_names)
    text = detok(tokens)
    spans = iob_to_spans(tokens, tags)
    for s in spans:
        raw = s["type"]
        s["type"] = _PICO_MAP.get(raw, raw)
    return {
        "id": example.get("id") or str(uuid.uuid4()),
        "dataset": "ebmnlp",
        "split": split,
        "text": text,
        "schema": "pico",
        "labels": spans
    }

def adapt_naver(example, split, tag_names=None):
    tokens = example["tokens"]
    tags = _decode_tags(example["ner_tags"], tag_names)
    if tags and isinstance(tags[0], str) and "_" in tags[0]:
        tags = _normalize_suffix_bio_fast(tags)
    text = " ".join(tokens)           # detok 간소화 (속도)
    spans = iob_to_spans(tokens, tags)
    exid = example.get("id")
    exid = str(exid if not isinstance(exid, list) else uuid.uuid4())
    return {"id": exid, "dataset": "naver", "split": split, "text": text, "schema": "generic", "labels": spans}