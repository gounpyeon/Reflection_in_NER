# prompts.py

# 데이터셋별 타입 → 간단 설명(한국어)
TYPE_GLOSSARY = {
    "conll": {
        "PER": "인물(사람 이름, 성/이름 포함)",
        "ORG": "조직/기관/회사/팀",
        "LOC": "지명(국가/도시/지역)",
        "MISC": "기타(국가명, 제품명 등 잡다한 고유명)"
    },
    "wnut": {
        "person": "인물명(계정명 포함 가능)",
        "location": "지명/장소",
        "group": "집단/커뮤니티/팀",
        "corporation": "기업/브랜드",
        "product": "제품/서비스명",
        "creative-work": "창작물(책/영화/노래/앱 등)"
    },
    # ebmnlp는 PIO 전용 템플릿을 따로 쓰므로 generic glossary는 생략 가능
    "naver": {
        "AFW": "예술/작품(책/영화/음반 등)",
        "ANM": "동물",
        "CVL": "민족/국민/시민",
        "DAT": "날짜(표기된 날짜/연월일)",
        "EVT": "사건/행사",
        "FLD": "학문/분야",
        "LOC": "지명/장소",
        "MAT": "물질/재료",
        "NUM": "수량/숫자",
        "ORG": "조직/기관",
        "PER": "인명",
        "PLT": "식물",
        "TIM": "시간(시각/기간)",
        "TRM": "전문용어/용어"
    }
}

def _render_types(types):
    # 사람이 읽기 쉬운 쉼표 나열
    return ", ".join(str(t) for t in types)

def _render_type_guide(dataset: str, types: list[str]) -> str:
    """
    타입 목록에 맞춰 간단 정의를 bullet로 생성.
    glossary에 없는 타입은 이름만 표시.
    """
    g = TYPE_GLOSSARY.get(dataset, {})
    lines = []
    for t in types or []:
        desc = g.get(t, "")
        if desc:
            lines.append(f"- {t}: {desc}")
        else:
            lines.append(f"- {t}")
    return "\n".join(lines)


# 기존 GENERIC_INSTRUCTION를 다음으로 교체
GENERIC_INSTRUCTION = """You are a precise NER extractor.
Extract all entity mentions with their types from the sentence.
Use ONLY the following entity types: {type_list}

Short definitions (for disambiguation):
{type_defs}

Return JSON ONLY in this shape:
{{"entities":[{{"text":"...","type":"..."}}]}}
Sentence: "{text}"
"""

# PIO 템플릿은 그대로 두되 살짝 보강 (선택)
PIO_INSTRUCTION = """You are a biomedical PIO extractor for NER.
Categories to extract (use JSON keys P/I/O):
- P = Population (PAR): 연구 대상/집단
- I = Intervention (INT): 처치/중재/비교군 포함 가능
- O = Outcome (OUT): 결과/지표/효과

Return JSON ONLY in this shape:
{{"P": ["..."], "I": ["..."], "O": ["..."]}}
If a category has no spans, return an empty list.
Text: "{text}"
"""

def build_prompt(sample):
    ds = sample.get("dataset", "")
    if sample["schema"] == "generic":
        tlist = sample.get("type_vocab") or []
        guide = _render_type_guide(ds, tlist)

        # 기본 generic 메시지
        base = GENERIC_INSTRUCTION.format(
            text=sample["text"],
            type_list=_render_types(tlist) if tlist else "PER, ORG, LOC, MISC",
            type_defs=guide if guide else "- PER, ORG, LOC, MISC"
        )

        # 네이버: 한국어 공백 토크나이즈 규칙을 추가 (연결/분할 금지)
        if ds == "naver":
            base += """
IMPORTANT (Korean whitespace tokenization):
- Select entity spans as contiguous sequences of whitespace-delimited tokens.
- Do NOT split or merge tokens. The "text" must be exactly the tokens joined by a single space.
"""
        return base
    else:
        # pico는 P/I/O 전용
        return PIO_INSTRUCTION.format(text=sample["text"])

# gold-free critic (정답 안 봄) — 입력 문장도 넣어줌
CRITIC_TEMPLATE = """You are a meticulous NER reviewer.

# Input text
{text}

# Current prediction
{pred}

# Task
Based ONLY on the input text and the current prediction above,
identify likely issues and propose actionable fixes:

- Boundary errors (too short/long, off-by-one)
- Type errors (wrong entity type)
- Spurious entities (should be removed)
- Missing entities (should be added)

Explain briefly WHY (one line each), then list concrete edits as bullets:
- ADD: "text" -> TYPE
- REMOVE: "text"
- RETAG: "text" -> NEW_TYPE

Keep it concise. Do NOT assume access to gold labels.
"""