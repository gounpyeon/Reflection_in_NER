# cache_unified.py
from __future__ import annotations
import os, hashlib, pickle, json, time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datasets import load_dataset, load_from_disk

# 너의 기존 함수 그대로 사용
from dataloader_unified import build_unified_corpus

DEFAULT_PATHS = {
    "conll":  "/root/2025_HCLT/data/hgissbkh___conll2003-en",
    "wnut":   "/root/2025_HCLT/data/dany0407___wnut_17",
    "ebmnlp": "/root/2025_HCLT/data/reginaboateng___ebmnlp_pico",
    "naver":  "/root/2025_HCLT/data/naver_ner",  # load_from_disk 디렉토리
}

def iob_to_spans_from_ids(tokens, tag_ids, id2label):
    spans, cur, cur_type = [], [], None
    def flush():
        nonlocal cur, cur_type
        if cur and cur_type:
            spans.append({"text": " ".join(cur).strip(), "type": cur_type})
        cur, cur_type = [], None

    for tok, tid in zip(tokens, tag_ids):
        lab = id2label[int(tid)]
        if lab == "O":
            flush()
        elif lab.startswith("B-"):
            flush()
            cur_type = lab[2:]
            cur = [tok]
        elif lab.startswith("I-"):
            t = lab[2:]
            if cur and cur_type == t:
                cur.append(tok)
            else:
                flush()
                cur_type = t
                cur = [tok]
        else:
            flush()
    flush()
    return spans

def naver_tags_to_spans(tokens, tag_str_list):
    spans, cur, cur_type = [], [], None
    def flush():
        nonlocal cur, cur_type
        if cur and cur_type:
            spans.append({"text": " ".join(cur).strip(), "type": cur_type})
        cur, cur_type = [], None

    for tok, t in zip(tokens, tag_str_list):
        if not isinstance(t, str) or t in ("O", "-") or "_" not in t:
            flush(); continue
        etype, pos = t.split("_", 1)
        pos = pos.upper()
        if pos == "B":
            flush(); cur_type = etype; cur = [tok]
        elif pos == "I":
            if cur and cur_type == etype:
                cur.append(tok)
            else:
                flush(); cur_type = etype; cur = [tok]
        else:
            flush()
    flush()
    return spans

def ebm_to_pio_spans(tokens, tag_ids, id2label):
    spans, cur, cur_raw = [], [], None  # raw: PAR/INT/OUT
    def flush():
        nonlocal cur, cur_raw
        if cur and cur_raw:
            map_ = {"PAR": "P", "INT": "I", "OUT": "O"}
            t = map_.get(cur_raw)
            if t:
                spans.append({"text": " ".join(cur).strip(), "type": t})
        cur, cur_raw = [], None

    for tok, tid in zip(tokens, tag_ids):
        lab = id2label[int(tid)]
        if lab == "O":
            flush()
        elif lab.startswith("I-"):
            raw = lab[2:]
            if cur and cur_raw == raw:
                cur.append(tok)
            else:
                flush(); cur_raw = raw; cur = [tok]
        else:
            flush()
    flush()
    return spans

def _dir_digest(path: str) -> Tuple[int, float]:
    """
    path 하위 모든 파일의 총 size, 가장 최근 mtime을 반환.
    파일 수가 많아도 빠르게 동작(내용 해시 X).
    """
    total_size = 0
    latest_mtime = 0.0
    p = Path(path)
    if not p.exists():
        return (0, 0.0)
    if p.is_file():
        st = p.stat()
        return (st.st_size, st.st_mtime)

    for root, _, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            try:
                st = fp.stat()
                total_size += st.st_size
                if st.st_mtime > latest_mtime:
                    latest_mtime = st.st_mtime
            except FileNotFoundError:
                # 탐색 중 파일이 사라진 경우 스킵
                pass
    return (total_size, latest_mtime)

def _sources_fingerprint(paths: Dict[str, str], splits, extra_salt: str = "") -> str:
    """
    각 소스 디렉토리의 (path, total_size, latest_mtime)를 모아 sha1 해시로 만듦.
    소스 내용이 바뀌면 fingerprint가 달라져 캐시 무효화됨.
    """
    parts: List[str] = [f"SPLITS::{','.join(splits)}"]
    for k in sorted(paths.keys()):
        p = paths[k]
        size, mtime = _dir_digest(p)
        parts.append(f"{k}::{p}::{size}::{mtime}")
    if extra_salt:
        parts.append(f"SALT::{extra_salt}")
    h = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return h

def _extract_types_from_iob_names(names):
    """
    ['O','B-PER','I-PER', ...] 또는 ['O','B-person','I-person', ...]에서
    고유 타입만 뽑아 ['PER','ORG',...] 혹은 ['person','location',...] 반환
    """
    types = set()
    for n in names:
        if isinstance(n, str) and (n.startswith("B-") or n.startswith("I-")):
            t = n[2:]
            types.add(t)
    return sorted(types)

def _naver_types_from_tags(ds, max_scan=1000):
    """
    ds: HuggingFace Dataset (naver['test'])
    앞에서 max_scan개 샘플만 훑어서 'AFW_B' → 'AFW' 식으로 타입 접두만 수집.
    """
    seen = set()
    for i, ex in enumerate(ds):
        if i >= max_scan:
            break
        tags = ex.get("ner_tags") or []
        for tag in tags:
            if isinstance(tag, str) and "_" in tag:
                seen.add(tag.split("_", 1)[0])
    return sorted(seen)


def build_unified_corpus_cached(
    paths: Optional[Dict[str, str]] = None,
    cache_dir: str = ".cache_unified",
    force_rebuild: bool = False,
    splits: tuple = ("test",),
    extra_salt: str = "",     # 스키마/코드가 바뀌었을 때 버전 업용
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    1) 소스 디렉토리 fingerprint 계산
    2) cache_dir/unified_{fp}.pkl 있으면 로드
    3) 없으면 build_unified_corpus() 실행 후 저장
    """
    paths = paths or DEFAULT_PATHS
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    fp = _sources_fingerprint(paths, splits, extra_salt=extra_salt)
    cache_bin = Path(cache_dir) / f"unified_{fp}.pkl"
    manifest = Path(cache_dir) / "manifest.json"

    if cache_bin.exists() and not force_rebuild:
        with open(cache_bin, "rb") as f:
            uni = pickle.load(f)
        # manifest에 최근 기록 남기기(선택)
        _write_manifest(manifest, fp, paths, hit=True)
        return uni

    # 없거나 강제 재생성이면 새로 빌드
    uni = {}

    # 1) CoNLL

    conll = load_dataset("/root/2025_HCLT/data/hgissbkh___conll2003-en")
    id2label_conll = conll["test"].features["ner"].feature.names    # ['O','B-PER',...]
    type_vocab_conll = sorted({lab[2:] for lab in id2label_conll if lab.startswith(("B-","I-"))})
    uni["conll"] = {}
    for split in ["train","validation","test"]:
        if split not in conll: continue
        uni["conll"][split] = []
        for ex in conll[split]:
            tokens = ex["words"]
            tag_ids = ex["ner"]                 # ← 중요: ‘ner’ 필드
            spans = iob_to_spans_from_ids(tokens, tag_ids, id2label_conll)
            uni["conll"][split].append({
                "dataset": "conll",
                "split": split,
                "schema": "generic",
                "text": " ".join(tokens),
                "labels": spans,                # ★ 스팬 딕셔너리 리스트
                "type_vocab": type_vocab_conll,
            })

    # 2) WNUT
    wnut = load_dataset("/root/2025_HCLT/data/dany0407___wnut_17")
    id2label_wnut = wnut["test"].features["ner_tags"].feature.names
    type_vocab_wnut = sorted({lab[2:] for lab in id2label_wnut if lab.startswith(("B-","I-"))})
    uni["wnut"] = {}
    for split in ["train","validation","test"]:
        if split not in wnut: continue
        uni["wnut"][split] = []
        for ex in wnut[split]:
            tokens = ex["tokens"]
            tag_ids = ex["ner_tags"]           # ← 중요
            spans = iob_to_spans_from_ids(tokens, tag_ids, id2label_wnut)
            uni["wnut"][split].append({
                "dataset": "wnut",
                "split": split,
                "schema": "generic",
                "text": " ".join(tokens),
                "labels": spans,
                "type_vocab": type_vocab_wnut,
            })

    # 3) EBMNLP  (P/I/O로 평가하므로 안내는 P=PAR, I=INT, O=OUT로 매핑해 제공)
    ebm = load_dataset("/root/2025_HCLT/data/reginaboateng___ebmnlp_pico")
    id2label_ebm = ebm["test"].features["ner_tags"].feature.names  # ['O','I-INT','I-OUT','I-PAR']
    type_vocab_ebm = ["PAR (Population)", "INT (Intervention)", "OUT (Outcome)"]
    uni["ebmnlp"] = {}
    for split in ["train","validation","test"]:
        if split not in ebm: continue
        uni["ebmnlp"][split] = []
        for ex in ebm[split]:
            tokens = ex["tokens"]
            # 잡음 라인(예: ['DOCSTART-'])는 스킵 권장
            if len(tokens) == 1 and tokens[0].startswith("DOCSTART"):
                continue
            tag_ids = ex["ner_tags"]
            spans = ebm_to_pio_spans(tokens, tag_ids, id2label_ebm)
            uni["ebmnlp"][split].append({
                "dataset": "ebmnlp",
                "split": split,
                "schema": "pico",               # 프롬프트는 P/I/O 키 사용
                "text": " ".join(tokens),
                "labels": spans,                # {"text":..., "type": "P|I|O"}
                "type_vocab": type_vocab_ebm,
            })

    # 4) NAVER
    naver = load_from_disk("/root/2025_HCLT/data/naver_ner")
    # 타입 목록 샘플링해서 뽑기
    def _naver_types(ds, max_scan=1000):
        """
        ds: datasets.Dataset (ex: naver["test"])
        앞에서부터 최대 max_scan개 샘플을 '샘플 단위'로 순회하며
        'PER_B' 형태 태그의 접두부(PER 등)만 수집.
        """
        seen = set()
        n = min(max_scan, len(ds))
        # Dataset을 그대로 슬라이스(dict 아님)하는 안전한 방법
        for ex in ds.select(range(n)):
            tags = ex["ner_tags"]  # list[str]
            for t in tags:
                if isinstance(t, str) and "_" in t:
                    seen.add(t.split("_", 1)[0])
        return sorted(seen)

    type_vocab_naver = _naver_types(naver["test"])
    uni["naver"] = {}
    for split in ["train","validation","test"]:
        if split not in naver: continue
        uni["naver"][split] = []
        # ⚠️ 반드시 per-example 순회: for ex in naver[split]
        for ex in naver[split]:
            tokens = ex["tokens"]
            tag_strs = ex["ner_tags"]          # ["PER_B","PER_I","O",...]
            spans = naver_tags_to_spans(tokens, tag_strs)
            uni["naver"][split].append({
                "dataset": "naver",
                "split": split,
                "schema": "generic",
                "text": " ".join(tokens),
                "labels": spans,
                "type_vocab": type_vocab_naver,
            })

    # 저장
    with open(cache_bin, "wb") as f:
        pickle.dump(uni, f, protocol=pickle.HIGHEST_PROTOCOL)
    _write_manifest(manifest, fp, paths, hit=False)
    return uni

def _write_manifest(manifest_path: Path, fp: str, paths: Dict[str,str], hit: bool):
    rec = {
        "fingerprint": fp,
        "paths": paths,
        "cache_file": f"unified_{fp}.pkl",
        "cache_dir": str(manifest_path.parent),
        "ts": time.time(),
        "cache_hit": hit,
    }
    # 여러 기록을 남기고 싶으면 리스트 append, 단건이면 덮어쓰기
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

# build_unified_corpus_cached(force_rebuild=True)