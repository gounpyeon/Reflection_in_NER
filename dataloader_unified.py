# dataloader_unified.py
import os, time
from typing import Iterator, Dict, Any, List, Tuple
from datasets import load_dataset, load_from_disk
from build_corpus import convert_datasetdict

def build_unified_corpus(splits: Tuple[str, ...] = ("test",), naver_num_proc: int = 0):
    print(f"[INFO] target splits: {splits}")

    # 가능하면 split= 로 바로 불러오기
    conll  = {"test": load_dataset("./data/hgissbkh___conll2003-en", split="test")}
    wnut   = {"test": load_dataset("./data/dany0407___wnut_17", split="test")}
    ebmnlp = {"test": load_dataset("./data/reginaboateng___ebmnlp_pico", split="test")}
    naver  = load_from_disk("./data/naver_ner")  # DatasetDict (train/valid/test)

    uni = {}
    for name, ds in [("conll", conll), ("wnut", wnut), ("ebmnlp", ebmnlp), ("naver", naver)]:
        print(f"\n[START] {name}")
        t0 = time.perf_counter()
        uni[name] = convert_datasetdict(ds, name, splits=splits)
        print(f"[DONE] {name} took {time.perf_counter()-t0:.2f}s")
    return uni

def iter_samples(unified: Dict[str, Dict[str, List[Dict]]], splits=("test",)) -> Iterator[Dict[str,Any]]:
    for dname, dd in unified.items():
        for sp in splits:
            for rec in dd.get(sp, []):
                yield rec