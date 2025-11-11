# llm_clients.py
from __future__ import annotations
from typing import Literal, Optional, Dict, Any
import os, requests, time

class BaseLLM:
    def chat(self, prompt: str, role: Literal["extractor","critic"]="extractor") -> Dict[str, Any]:
        raise NotImplementedError

def make_llm(kind: str, **kw) -> 'BaseLLM':
    kind = kind.lower()
    if kind == "openrouter":
        return OpenRouterClient(**kw)
    if kind == "ollama":
        return OllamaClient(**kw)
    raise ValueError(f"Unknown LLM kind: {kind}")

# =============================
# OpenRouter (OpenAI 호환 API)
# =============================
class OpenRouterClient(BaseLLM):
    def __init__(self,
                 model_extractor: str = "gpt-4o-mini",
                 model_critic: str = "gpt-4o-mini",
                 temperature: float = 0.0,
                 api_key: str = None,
                 base_url: str = "https://openrouter.ai/api/v1"):
        
        from openai import OpenAI
        if not api_key: raise ValueError("API key required for OpenRouterClient")

        self.cli = OpenAI(base_url=base_url,
                          api_key=api_key)
        self.m = {"extractor": model_extractor, "critic": model_critic}
        self.t = temperature

    def chat(self, prompt: str, role: Literal["extractor","critic"]="extractor") -> Dict[str, Any]:
        start = time.time()
        r = self.cli.chat.completions.create(
            model=self.m[role], temperature=self.t,
            messages=[
                {"role": "system", "content": "You are a careful IE model. Respond in JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        elapsed = time.time() - start

        # 텍스트
        out_text = r.choices[0].message.content.strip() if r.choices else ""

        # 토큰 사용량 (OpenAI 호환)
        in_tok = None
        out_tok = None
        usage = getattr(r, "usage", None)
        if usage is not None:
            # usage는 보통 SimpleNamespace처럼 attribute 접근 가능
            in_tok  = getattr(usage, "prompt_tokens", None)
            out_tok = getattr(usage, "completion_tokens", None)

        return {
            "text": out_text,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "latency": elapsed
        }

# =============================
# Ollama (로컬 서버, 기본 포트 11434)
# =============================
class OllamaClient(BaseLLM):
    def __init__(self,
                 model_extractor: str = "llama3.1:8b",
                 model_critic: str = "llama3.1:8b",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.0):
        self.rq = requests
        self.host = host.rstrip("/")
        self.m = {"extractor": model_extractor, "critic": model_critic}
        self.t = temperature

    def chat(self, prompt: str, role: Literal["extractor","critic"]="extractor") -> Dict[str, Any]:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.m[role],
            "messages": [
                {"role": "system", "content": "You are a careful IE model. Respond in JSON."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": self.t, "num_threads": 8}
        }

        start = time.time()
        res = self.rq.post(url, json=payload, timeout=600)
        elapsed_wall = time.time() - start
        res.raise_for_status()
        j = res.json()

        # 텍스트
        out_text = (j.get("message", {}) or {}).get("content", "").strip()

        # Ollama 메트릭 매핑
        # - prompt_eval_count: 프롬프트(입력) 토큰 수 비슷한 의미
        # - eval_count: 생성(출력) 토큰 step 수
        # - total_duration: ns 단위 총 소요 시간
        in_tok  = j.get("prompt_eval_count")
        out_tok = j.get("eval_count")

        latency = None
        if "total_duration" in j and isinstance(j["total_duration"], (int, float)):
            # ns -> s
            latency = float(j["total_duration"]) / 1e9
        else:
            # fallback: 벽시계 시간
            latency = elapsed_wall

        return {
            "text": out_text,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "latency": latency
        }