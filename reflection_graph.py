from __future__ import annotations
from typing import TypedDict, Literal, List, Dict, Any, Optional, Tuple
import json, re, uuid
from copy import deepcopy
import traceback


# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# 평가 함수
from evaluate import prf1
from prompts import build_prompt, CRITIC_TEMPLATE
from llm_clients import make_llm


# =========================
# 디버깅 헬퍼
# =========================
def safe_try(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        print("[DEBUG] Exception:", repr(e))
        traceback.print_exc()
        raise

# =========================
# LLM 클라이언트 어댑터
# =========================

class LLMClient:
    """
    교체 가능 어댑터. OpenAI 예시.
    필요 시 __init__에서 다른 공급자 클라이언트로 교체.
    """
    def __init__(self, model_extractor: str = "gpt-4o-mini",
                       model_critic: str = "gpt-4o-mini",
                       temperature: float = 0.0):
        
        from openai import OpenAI
        self.client = OpenAI()
        self.model_extractor = model_extractor
        self.model_critic = model_critic
        self.temperature = temperature

    def chat(self, prompt: str, role: Literal["extractor","critic"]="extractor") -> str:
        model = self.model_extractor if role == "extractor" else self.model_critic
        resp = self.client.chat.completions.create(
            model=model,
            temperature=self.temperature,
            messages=[{"role":"system","content":"You are a careful information extraction model."},
                      {"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content.strip()


# =========================
# JSON 유틸/파서
# =========================

def try_json_loads(s: str) -> Optional[dict]:
    """
    - 코드블럭, 프리텍스트 섞여도 JSON만 뽑아보는 느슨한 파서.
    - 첫 번째 {...} or [...] 블럭만 파싱 시도.
    """
    # 코드블럭 제거
    s = re.sub(r"```(json)?", "", s).strip()
    # 가장 바깥 JSON 블럭 추정
    m = re.search(r"(\{.*\}|$begin:math:display$.*$end:math:display$)", s, re.S)
    cand = m.group(1) if m else s
    try:
        return json.loads(cand)
    except Exception:
        return None


def to_generic_pred(schema: str, llm_out: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    - schema == 'generic':
        {"entities":[{"text":"...","type":"..."}]}
    - schema == 'pico' (PIO만 사용):
        {"P":[...], "I":[...], "O":[...]}  -> [{"text":..., "type":"P"|"I"|"O"}]
      * 모델이 실수로 "C"를 내더라도 무시
    """
    if not isinstance(llm_out, dict):
        return []

    if schema == "generic":
        ents = llm_out.get("entities", [])
        out = []
        if isinstance(ents, list):
            for e in ents:
                if isinstance(e, dict) and "text" in e and "type" in e:
                    txt = str(e["text"]).strip()
                    typ = str(e["type"]).strip()
                    if txt:
                        out.append({"text": txt, "type": typ})
        return out

    # pico → P/I/O만
    out = []
    # 키 대소문자 섞임을 방지
    norm = {str(k).upper(): v for k, v in llm_out.items() if isinstance(k, str)}
    for t in ("P", "I", "O"):
        vals = norm.get(t, []) or []
        if isinstance(vals, list):
            for s in vals:
                txt = str(s).strip()
                if txt:
                    out.append({"text": txt, "type": t})
    # 모델이 실수로 "C"를 줘도 무시됨
    return out


def normalize_text(t: str) -> str:
    t = t.strip()
    t = t.strip('"\''"“”‘’")
    t = re.sub(r"\s+", " ", t)
    return t.lower()


# =========================
# LangGraph 상태 정의
# =========================

class RState(TypedDict):
    sample: Dict[str, Any]              # {"text", "labels", "schema", ...}
    round_idx: int                      # 현재 라운드 (0 시작)
    max_rounds: int                     # 최대 라운드
    pred_history: List[Dict[str, Any]]  # 라운드별 기록(예측/토큰/시간/피드백)
    pred_current: List[Dict]            # 현재 라운드 예측
    feedback: Optional[str]             # critic 피드백(자연어)

# =========================
# 노드 구현
# =========================

class ReflectionNodes:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extractor(self, state: RState) -> RState:
        sample = state["sample"]
        allowed = sample.get("type_vocab")  # ← 추가
        # raw = self.llm.chat(build_prompt(sample), role="extractor")
        raw = safe_try(self.llm.chat, build_prompt(sample), role="extractor")
        js = try_json_loads(raw["text"]) or {}
        pred = to_generic_pred(sample["schema"], js)
        state["pred_current"] = pred
        state["pred_history"].append({
            "stage": "extract",
            "pred": pred,
            "in_tokens": raw.get("in_tokens"),
            "out_tokens": raw.get("out_tokens"),
            "latency": raw.get("latency"),
        })
        return state

    def critic(self, state: RState) -> RState:
        pred = deepcopy(state["pred_current"])
        text = state["sample"]["text"]
        # fb_raw = self.llm.chat(CRITIC_TEMPLATE.format(text=text, pred=pred), role="critic")
        fb_raw = safe_try(self.llm.chat, CRITIC_TEMPLATE.format(text=text, pred=pred), role="critic")
        fb_text = fb_raw.get("text", "")
        state["feedback"] = fb_text

        # (선택) critic 호출 자체도 기록하고 싶으면:
        state["pred_history"].append({
            "stage": "critic",
            "feedback": fb_text,
            "in_tokens": fb_raw.get("in_tokens"),
            "out_tokens": fb_raw.get("out_tokens"),
            "latency": fb_raw.get("latency"),
        })
        return state

    def reviser(self, state: RState) -> RState:
        sample = state["sample"]
        allowed = sample.get("type_vocab")  # ← 추가
        base = build_prompt(sample)
        feedback = state.get("feedback") or ""
        prompt = f"{base}\n\n# Review Feedback\n{feedback}\n\nRe-extract with corrections. Return only JSON."
        # raw = self.llm.chat(prompt, role="extractor")
        raw = safe_try(self.llm.chat, prompt, role="extractor")
        js = try_json_loads(raw["text"]) or {}
        pred = to_generic_pred(sample["schema"], js)
        state["pred_current"] = pred
        state["pred_history"].append({
            "stage": "revise",
            "pred": pred,
            "used_feedback": feedback,           # ← 이번 라운드에 사용한 피드백 저장
            "in_tokens": raw.get("in_tokens"),
            "out_tokens": raw.get("out_tokens"),
            "latency": raw.get("latency"),
        })

        state["round_idx"] += 1
        return state


# =========================
# 그래프 구성
# =========================

def build_reflection_graph(llm: Optional[LLMClient] = None):
    llm = llm or LLMClient()
    nodes = ReflectionNodes(llm)
    graph = StateGraph(RState)

    graph.add_node("extract", nodes.extractor)
    graph.add_node("critic",  nodes.critic)
    graph.add_node("revise",  nodes.reviser)

    # START → extract (초기 1회) → critic → revise → (critic로 루프 or 종료)
    graph.add_edge(START,     "extract")
    graph.add_edge("extract", "critic")
    graph.add_edge("critic",  "revise")

    def should_continue(state: RState) -> Literal["loop","stop"]:
        return "stop" if state["round_idx"] >= state["max_rounds"] else "loop"

    graph.add_conditional_edges(
        "revise",
        should_continue,
        {"loop": "critic", "stop": END},   # ← loop 대상만 critic으로 변경
    )
    return graph.compile()

def build_reflection_graph_from_cfg(cfg: dict):
    llm = make_llm(**cfg["llm"])
    return build_reflection_graph(llm)

# =========================
# 편의 실행 함수
# =========================

def run_reflection_rounds_langgraph(
    sample: Dict[str, Any],
    llm: Optional[Any] = None,
    llm_kind: Optional[str] = None,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    rounds: int = 10,
    with_type: bool = True,      # ← 추가
) -> Dict[str, Any]:
    if llm is None:
        if llm_kind is None:
            raise ValueError("Provide either `llm` or (`llm_kind` + `llm_kwargs`).")
        llm = make_llm(llm_kind, **(llm_kwargs or {}))

    app = build_reflection_graph(llm)
    init_state: RState = {
        "sample": sample,
        "round_idx": 0,
        "max_rounds": rounds,
        "pred_history": [],
        "pred_current": [],
        "feedback": None,
    }
    final_state = app.invoke(init_state, config={"recursion_limit": 2*rounds + 5})
    history = final_state["pred_history"]

    # === 사후 평가 (gold는 여기서만 사용) ===
    gold = sample.get("labels", []) or []
    best_f, best_idx = -1.0, -1
    best_pred = []
    for i, h in enumerate(history):
        pred_i = h.get("pred", [])
        # critic 단계는 pred가 없을 수 있음
        if pred_i is None:
            continue
        _, _, f = prf1(gold, pred_i, with_type=with_type)
        if f > best_f:
            best_f = f
            best_idx = i
            best_pred = pred_i

    best_score = (0.0, 0.0, 0.0)
    if best_idx >= 0:
        p, r, f = prf1(gold, best_pred, with_type=with_type)
        best_score = (p, r, f)

    return {
        "best_pred": best_pred,
        "best_score": best_score,
        "history": history,
    }