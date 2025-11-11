# main_runner.py
from __future__ import annotations
import traceback
import os, csv, json, uuid, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from cache_unified import build_unified_corpus_cached
from reflection_graph import run_reflection_rounds_langgraph
from llm_clients import make_llm
from evaluate import prf1
from tqdm.auto import tqdm

import random
import time

# ---------- 유틸 ----------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def now_kst_stamp() -> str:
    # Asia/Seoul 기준 타임스탬프 (파일명에 안전)
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")

def norm_tuple_list(ents: List[Dict], with_type=True):
    import re
    def _norm(s: str):
        s = s.strip().strip('"\'""''')
        s = re.sub(r"\s+", " ", s)
        return s.lower()
    out = []
    for e in ents:
        txt = _norm(e["text"])
        typ = e.get("type","") if with_type else ""
        out.append((txt, typ))
    return set(out)

def micro_counts(gold: List[Dict], pred: List[Dict], with_type=True) -> Tuple[int,int,int]:
    G = norm_tuple_list(gold, with_type)
    P = norm_tuple_list(pred, with_type)
    tp = len(G & P)
    fp = len(P - G)
    fn = len(G - P)
    return tp, fp, fn

def micro_prf1(total_tp, total_fp, total_fn):
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
    return prec, rec, f1

# ---------- JSON 디코드 에러 대응 함수 ----------
def run_reflection_with_retry(sample, llm, rounds, with_type, max_retries=5):
    """JSON 디코드 에러 발생시 재시도하는 래퍼 함수"""
    for attempt in range(max_retries + 1):  # 0부터 max_retries까지
        try:
            result = run_reflection_rounds_langgraph(sample=sample, llm=llm, rounds=rounds, with_type=with_type)
            return result
        except json.JSONDecodeError as e:
            print(f"    [WARNING] JSON decode error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
            if attempt < max_retries:
                # 재시도 전 잠시 대기 (API 안정화)
                time.sleep(1.0 * (attempt + 1))  # 1초, 2초, 3초... 점진적 증가
                continue
            else:
                # 최대 재시도 초과시 placeholder 결과 반환
                print(f"    [ERROR] Max retries exceeded. Using placeholder result.")
                return {
                    "history": [{
                        "stage": "placeholder",
                        "pred": [],  # 빈 예측
                        "in_tokens": 0,
                        "out_tokens": 0,
                        "latency": 0.0
                    }]
                }
        except Exception as e:
            # 다른 에러의 경우에도 재시도 가능하도록
            print(f"    [WARNING] Other error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
            print(traceback.print_exc())
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            else:
                print(f"    [ERROR] Max retries exceeded for other error. Using placeholder result.")
                return {
                    "history": [{
                        "stage": "placeholder_error",
                        "pred": [],
                        "in_tokens": 0,
                        "out_tokens": 0,
                        "latency": 0.0
                    }]
                }

# ---------- 병렬 처리용 샘플 실행 함수 ----------
def process_sample(args_tuple):
    """각 샘플을 처리하는 함수 (ThreadPoolExecutor용)"""
    ex, llm, rounds, with_type = args_tuple
    
    # JSON 디코드 에러 대응 포함하여 실행
    result = run_reflection_with_retry(sample=ex, llm=llm, rounds=rounds, with_type=with_type)
    
    history = result.get("history", [])
    
    # iteration별 사후 평가
    iter_metrics = []
    for i, h in enumerate(history):
        pred_i = h.get("pred", [])
        p_i, r_i, f_i = prf1(ex["labels"], pred_i, with_type=with_type)
        iter_metrics.append({"round": i, "precision": p_i, "recall": r_i, "f1": f_i})
    
    # 토큰/시간 합계
    total_in_tok  = sum((h.get("in_tokens")  or 0) for h in history)
    total_out_tok = sum((h.get("out_tokens") or 0) for h in history)
    total_latency = sum((h.get("latency")    or 0.0) for h in history)
    
    return {
        "sample": ex,
        "result": result,
        "iter_metrics": iter_metrics,
        "token_totals": {
            "in_tokens": total_in_tok,
            "out_tokens": total_out_tok,
            "latency": total_latency
        }
    }

# ---------- OpenRouter 전용 모델 설정 ----------
def get_openrouter_model_cfgs() -> List[Dict[str, Any]]:
    """OpenRouter API만 사용하는 모델 설정"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in environment")
    friendli_key = os.getenv("FRIENDLI_TOKEN")
    
    return [
        # --- Large Models ---
        {"name": "openrouter_gpt4omini",
         "llm": {"kind": "openrouter",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_key": api_key,
                 "model_extractor": "openai/gpt-4o-mini",
                 "model_critic": "openai/gpt-4o-mini",
                 "temperature": 0.0}},
        {"name": "openrouter_llama4scout",
         "llm": {"kind": "openrouter",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_key": api_key,
                 "model_extractor": "meta-llama/llama-4-scout",
                 "model_critic": "meta-llama/llama-4-scout",
                 "temperature": 0.0}},

        # --- Medium Models ---
        {"name": "openrouter_exaone4_32b",
         "llm": {"kind": "openrouter",
                 "base_url": "https://api.friendli.ai/serverless/v1",
                 "api_key": friendli_key,
                 "model_extractor": "LGAI-EXAONE/EXAONE-4.0.1-32B",
                 "model_critic": "LGAI-EXAONE/EXAONE-4.0.1-32B",
                 "temperature": 0.0}},
        {"name": "openrouter_gptoss20b",
         "llm": {"kind": "openrouter",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_key": api_key,
                 "model_extractor": "openai/gpt-oss-20b",
                 "model_critic": "openai/gpt-oss-20b",
                 "temperature": 0.0}},
        
        # --- Small Models ---
        {"name": "openrouter_llama3_8b",
         "llm": {"kind": "openrouter",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_key": api_key,
                 "model_extractor": "meta-llama/llama-3.1-8b-instruct",
                 "model_critic": "meta-llama/llama-3.1-8b-instruct",
                 "temperature": 0.0}},
        {"name": "openrouter_nemotron_9b",
         "llm": {"kind": "openrouter",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_key": api_key,
                 "model_extractor": "nvidia/nemotron-nano-9b-v2",
                 "model_critic": "nvidia/nemotron-nano-9b-v2",
                 "temperature": 0.0}},
    ]

# ---------- 메인 실행 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="reflection rounds")
    parser.add_argument("--with_type", action="store_true", help="type-aware evaluation (default: False)")
    parser.add_argument("--max_per_split", type=int, default=200, help="각 split에서 평가할 최대 샘플 수")
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling")
    parser.add_argument("--datasets", type=str, default="conll,wnut,ebmnlp,naver",
                        help="콤마로 구분: conll,wnut,ebmnlp,naver")
    parser.add_argument("--splits", type=str, default="test",
                        help="콤마로 구분: train,validation,test")
    parser.add_argument("--model", type=str, default="", 
                        help="실행할 모델 이름(콤마로 여러 개 가능). --list-models로 가능한 이름 확인.")
    parser.add_argument("--list-models", action="store_true",
                        help="가능한 모델 이름 목록을 출력하고 종료")
    parser.add_argument("--select", type=str, choices=["best", "last"], default="best",
                        help="사후 평가에서 사용할 예측 선택 전략: best(라운드 중 최고 F1) | last(마지막 라운드)")
    parser.add_argument("--outdir", type=str, default="runs_ffinal",
                        help="결과 저장 디렉터리")
    parser.add_argument("--tag", type=str, default="", help="선택: 파일명/폴더명에 붙일 사용자 태그")
    parser.add_argument("--max_workers", type=int, default=8, help="병렬 처리용 최대 워커 수")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # 타임스탬프(폴더/파일 공통 suffix)
    ts = now_kst_stamp()
    run_root = Path(args.outdir) / "preds" / (f"{ts}" + (f"_{args.tag}" if args.tag else ""))
    ensure_dir(run_root)

    # 1) 데이터 로드 & 통합
    print("[*] Loading datasets ...")
    uni = build_unified_corpus_cached()
    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    selected_splits   = [s.strip() for s in args.splits.split(",") if s.strip()]

    # 2) OpenRouter 전용 모델 준비
    model_cfgs = get_openrouter_model_cfgs()
    
    # --list-models 처리
    if args.list_models:
        print("[*] Available OpenRouter models:")
        for m in model_cfgs:
            print(" -", m["name"])
        return
    
    # --model 선택 처리 (콤마 가능)
    selected_model_names = []
    if args.model:
        selected_model_names = [s.strip() for s in args.model.split(",") if s.strip()]

    if selected_model_names:
        name_set = set(selected_model_names)
        filtered = [m for m in model_cfgs if m["name"] in name_set]

        # 존재하지 않는 이름 경고
        all_names = set(m["name"] for m in model_cfgs)
        not_found = sorted(list(name_set - all_names))
        if not_found:
            print("[!] Unknown model name(s):", ", ".join(not_found))
            print("[*] Available models:")
            for m in model_cfgs:
                print(" -", m["name"])
            return

        model_cfgs = filtered
        print("[*] Selected models:", ", ".join(m["name"] for m in model_cfgs))
    else:
        print("[*] No --model given. Running ALL models.")
    
    # 3) 결과 CSV 준비 (전역 요약)
    summary_path = Path(args.outdir) / "summary.csv"
    is_new_summary = not summary_path.exists()
    
    # CSV 파일 접근용 lock
    csv_lock = threading.Lock()
    
    with open(summary_path, "a", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        if is_new_summary:
            sw.writerow(["ts","tag","model","dataset","split","with_type",
                         "rounds","n_samples","micro_precision","micro_recall","micro_f1","macro_f1"])

        # 4) 모델 루프
        for mcfg in tqdm(model_cfgs, desc="Models", position=0, dynamic_ncols=True):
            model_name = mcfg["name"]
            model_dir = run_root / model_name
            ensure_dir(model_dir)

            print(f"\n=== Model: {model_name} ===")
            llm = make_llm(**mcfg["llm"])

            # 5) 데이터셋 루프
            for dset in selected_datasets:
                if dset not in uni:
                    print(f"  (skip) dataset not found in unified: {dset}")
                    continue

                for split in selected_splits:
                    if split not in uni[dset] or len(uni[dset][split]) == 0:
                        print(f"  (skip) no samples: {dset}/{split}")
                        continue

                    random.seed(args.seed)

                    all_samples = uni[dset][split]
                    if len(all_samples) > args.max_per_split:
                        samples = random.sample(all_samples, args.max_per_split)
                    else:
                        samples = all_samples

                    print(f"  -> {dset}/{split} | eval {len(samples)} samples (max_workers={args.max_workers})")

                    # 파일 경로들 (타임스탬프 suffix 부착)
                    pred_path = model_dir / f"{model_name}__{dset}__{split}__{ts}.jsonl"
                    summary_json_path = model_dir / f"summary__{model_name}__{dset}__{split}__{ts}.json"

                    # ===== 병렬 처리로 샘플들 실행 =====
                    sample_results = []
                    
                    # 각 샘플과 필요한 인자들을 튜플로 묶기
                    tasks = [(ex, llm, args.rounds, args.with_type) for ex in samples]
                    
                    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                        # 모든 작업 제출
                        future_to_sample = {executor.submit(process_sample, task): task[0] for task in tasks}
                        
                        # 결과 수집 (tqdm으로 진행상황 표시)
                        for future in tqdm(
                            as_completed(future_to_sample),
                            total=len(tasks),
                            desc=f"{model_name} | {dset}/{split}",
                            leave=False,
                            position=1,
                            dynamic_ncols=True
                        ):
                            try:
                                result = future.result()
                                sample_results.append(result)
                            except Exception as exc:
                                sample = future_to_sample[future]
                                print(f"    [ERROR] Sample {sample.get('id', 'unknown')} generated exception: {exc}")
                                # 에러 발생 시 placeholder 결과 추가
                                sample_results.append({
                                    "sample": sample,
                                    "result": {"history": [{"stage": "error", "pred": [], "in_tokens": 0, "out_tokens": 0, "latency": 0.0}]},
                                    "iter_metrics": [{"round": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}],
                                    "token_totals": {"in_tokens": 0, "out_tokens": 0, "latency": 0.0}
                                })

                    # ===== 결과 처리 및 저장 =====
                    with open(pred_path, "w", encoding="utf-8") as fout:
                        total_tp = total_fp = total_fn = 0
                        macro_f1_sum = 0.0
                        n = 0
                        sum_in_tok = sum_out_tok = 0
                        sum_latency = 0.0

                        for sample_result in sample_results:
                            ex = sample_result["sample"]
                            result = sample_result["result"]
                            iter_metrics = sample_result["iter_metrics"]
                            token_totals = sample_result["token_totals"]
                            
                            history = result.get("history", [])

                            # 선택 전략: best(최고 F1) or last(마지막 라운드)
                            if args.select == "last":
                                chosen_idx = (len(history) - 1) if history else -1
                            else:  # "best"
                                chosen_idx = -1
                                best_f = -1.0
                                for m in iter_metrics:
                                    if m["f1"] > best_f:
                                        best_f = m["f1"]
                                        chosen_idx = m["round"]

                            chosen_pred = history[chosen_idx]["pred"] if (chosen_idx >= 0 and history) else []

                            # 누적 마이크로/매크로 집계 업데이트
                            p, r, f = prf1(ex["labels"], chosen_pred, with_type=args.with_type)
                            macro_f1_sum += f
                            n += 1

                            tp, fp, fn = micro_counts(ex["labels"], chosen_pred, with_type=args.with_type)
                            total_tp += tp; total_fp += fp; total_fn += fn

                            # 토큰/시간 누적
                            sum_in_tok += token_totals["in_tokens"]
                            sum_out_tok += token_totals["out_tokens"]
                            sum_latency += token_totals["latency"]

                            # per-sample JSONL 레코드 저장
                            rec = {
                                "id": ex.get("id", str(uuid.uuid4())),
                                "dataset": ex["dataset"],
                                "split": ex["split"],
                                "schema": ex["schema"],
                                "text": ex["text"],
                                "gold": ex["labels"],
                                "select": {"strategy": args.select, "selected_round": chosen_idx},
                                "pred": chosen_pred,
                                "iter_metrics": iter_metrics,
                                "history": history,
                                "token_totals": token_totals
                            }
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    # ---- 조합별 요약 저장 (JSON & CSV) ----
                    micro_p, micro_r, micro_f = micro_prf1(total_tp, total_fp, total_fn)
                    macro_f1 = (macro_f1_sum / n) if n > 0 else 0.0
                    
                    avg_in_tok  = (sum_in_tok  / n) if n else 0
                    avg_out_tok = (sum_out_tok / n) if n else 0
                    avg_latency = (sum_latency / n) if n else 0.0

                    combo_summary = {
                        "timestamp": ts,
                        "tag": args.tag,
                        "model": model_name,
                        "dataset": dset,
                        "split": split,
                        "with_type": bool(args.with_type),
                        "rounds": args.rounds,
                        "n_samples": n,
                        "metrics": {
                            "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
                            "macro": {"f1": macro_f1}
                        },
                        "avg_token_latency_per_sample": {
                            "in_tokens": avg_in_tok,
                            "out_tokens": avg_out_tok,
                            "latency_sec": avg_latency
                        },
                        "paths": {
                            "pred_jsonl": str(pred_path),
                            "summary_json": str(summary_json_path)
                        }
                    }
                    
                    # JSON 요약 저장 (모델×데이터셋×스플릿 단위)
                    with open(summary_json_path, "w", encoding="utf-8") as jf:
                        json.dump(combo_summary, jf, ensure_ascii=False, indent=2)

                    # 전역 CSV에 한 줄 (thread-safe)
                    with csv_lock:
                        sw.writerow([ts, args.tag, model_name, dset, split, int(args.with_type),
                                     args.rounds, n, f"{micro_p:.4f}", f"{micro_r:.4f}",
                                     f"{micro_f:.4f}", f"{macro_f1:.4f}"])
                        sf.flush()  # 즉시 파일에 쓰기
                    
                    print(f"    micro F1={micro_f:.4f}, macro F1={macro_f1:.4f} (P={micro_p:.3f}, R={micro_r:.3f})")
                    print(f"    saved: {pred_path.name}, {summary_json_path.name}")

    print(f"\n[*] Done. Summary CSV: {summary_path}")
    print(f"[*] Run folder: {run_root}")

if __name__ == "__main__":
    main()