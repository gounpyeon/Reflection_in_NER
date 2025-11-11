# build_corpus.py
import json, time, os
from datasets import Dataset, DatasetDict
from adapters import adapt_conll, adapt_wnut, adapt_ebmnlp, adapt_naver

def _get_tag_names(ds_split, field_name: str):
    feat = ds_split.features[field_name]
    if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
        return list(feat.feature.names)
    if hasattr(feat, "names"):
        return list(feat.names)
    return None

def convert_datasetdict(
    dd: DatasetDict,
    dataset_name: str,
    splits=("train","validation","test"),
    limit_per_split: int | None = None,
    num_proc: int = 0,
):
    out = {s: [] for s in splits}

    if dataset_name == "conll":
        fn = adapt_conll;  tag_field = "ner"
    elif dataset_name == "wnut":
        fn = adapt_wnut;   tag_field = "ner_tags"
    elif dataset_name == "ebmnlp":
        fn = adapt_ebmnlp; tag_field = "ner_tags"
    elif dataset_name == "naver":
        fn = adapt_naver;  tag_field = "ner_tags"
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    for split in splits:
        if split not in dd:
            continue
        ds_split = dd[split]
        if limit_per_split:
            ds_split = ds_split.select(range(min(limit_per_split, len(ds_split))))
        tag_names = _get_tag_names(ds_split, tag_field)
        n = len(ds_split)
        print(f"[CONVERT] {dataset_name}:{split} n={n} (tag_names={'ok' if tag_names else 'none'})")

        # ----- NAVER만 batched map(+병렬) 경로 -----
        if dataset_name == "naver":
                # ✅ NAVER: 순차 처리 (안정&충분히 빠름)
                out[split] = []
                for i, ex in enumerate(ds_split):
                    rec = adapt_naver(ex, split, tag_names=tag_names)
                    out[split].append(rec)
                    # if (i+1) % 1000 == 0 or (i+1) == n:
                        # print(f"  progress {i+1}/{n}")
                continue

        # ----- 그 외는 기존 단일 루프 또는 단건 map -----
        def _apply_single(ex):
            return {"_rec": fn(ex, split, tag_names=tag_names)}

        ds2: Dataset = ds_split.map(
            _apply_single,
            desc=f"{dataset_name}:{split}",
            load_from_cache_file=False,
            num_proc=None,  # 단일 프로세스 (이미 충분히 빠름)
        )
        out[split] = [r["_rec"] for r in ds2]

    return out

def dump_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")