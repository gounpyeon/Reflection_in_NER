# utils_iob.py
from typing import List, Tuple, Dict

# utils_iob.py (요지)
def iob_to_spans(tokens, tags):
    spans = []
    cur, cur_type = [], None
    def flush():
        nonlocal cur, cur_type
        if cur and cur_type:
            spans.append({"text": " ".join(cur), "type": cur_type})
        cur, cur_type = [], None

    for tok, tag in zip(tokens, tags):
        if not isinstance(tag, str): flush(); continue
        t = tag.strip().upper()
        if t == "O": flush(); continue
        if "-" not in t: flush(); continue
        prefix, ttype = t.split("-", 1)
        if prefix == "E": prefix = "I"
        if prefix == "S": prefix = "B"
        if prefix not in ("B","I"): flush(); continue

        if prefix == "B":
            flush(); cur = [tok]; cur_type = ttype
        else:  # I
            if cur_type == ttype: cur.append(tok)
            else: flush(); cur = [tok]; cur_type = ttype
    flush()
    return spans

# def iob_to_spans(tokens: List[str], tags: List[str]) -> List[Dict]:
#     """
#     IOB/IOB2/IOBES 모두 허용. B-XXX, I-XXX, E-XXX, S-XXX, O
#     반환: [{"type": "ORG", "text": "Apple Inc.", "span":[start,end]}] (end exclusive)
#     """
#     spans = []
#     cur_type, start, buf = None, None, []

#     def close(i):
#         if cur_type is None or start is None: 
#             return
#         text = " ".join(buf)
#         spans.append({"type": cur_type, "text": text, "span": [start, i]})

#     i = 0
#     while i < len(tokens):
#         tag = tags[i]
#         if tag == "O" or tag == "o":
#             # close current
#             if cur_type is not None:
#                 close(i)
#                 cur_type, start, buf = None, None, []
#             i += 1
#             continue

#         if "-" in tag:
#             prefix, ttype = tag.split("-", 1)
#             prefix = prefix.upper()
#             if prefix == "S":  # single
#                 if cur_type is not None:
#                     close(i)
#                     cur_type, start, buf = None, None, []
#                 spans.append({"type": ttype, "text": tokens[i], "span":[i, i+1]})
#                 i += 1
#                 continue

#             if prefix == "B":
#                 if cur_type is not None:
#                     # 기존 span 종료
#                     close(i)
#                 cur_type, start, buf = ttype, i, [tokens[i]]
#                 i += 1
#                 # I/E가 이어지면 계속 붙임
#                 while i < len(tokens):
#                     tag2 = tags[i]
#                     if "-" in tag2:
#                         p2, t2 = tag2.split("-",1)
#                         p2 = p2.upper()
#                         if (p2 in ("I","E")) and (t2 == cur_type):
#                             buf.append(tokens[i])
#                             if p2 == "E":
#                                 close(i+1)
#                                 cur_type, start, buf = None, None, []
#                             i += 1
#                             if p2 == "E":
#                                 break
#                             continue
#                     break
#                 continue

#             if prefix == "I":
#                 # 잘못된 I 시작 → 새 span으로 보정
#                 if cur_type is None:
#                     cur_type, start, buf = ttype, i, [tokens[i]]
#                 else:
#                     if ttype == cur_type:
#                         buf.append(tokens[i])
#                     else:
#                         # 타입 바뀌면 종료 후 새로 시작
#                         close(i)
#                         cur_type, start, buf = ttype, i, [tokens[i]]
#                 i += 1
#                 continue

#             if prefix == "E":
#                 if cur_type is None:
#                     # 보정: 단독 엔티티로 처리
#                     spans.append({"type": ttype, "text": tokens[i], "span":[i, i+1]})
#                 else:
#                     if ttype == cur_type:
#                         buf.append(tokens[i])
#                         close(i+1)
#                         cur_type, start, buf = None, None, []
#                     else:
#                         # 타입 불일치 → 이전 종료 후 단독 처리
#                         close(i)
#                         spans.append({"type": ttype, "text": tokens[i], "span":[i, i+1]})
#                 i += 1
#                 continue

#         else:
#             # 태그가 이상한 경우 O로 취급
#             if cur_type is not None:
#                 close(i)
#                 cur_type, start, buf = None, None, []
#             i += 1

#     # 문장 끝에 열린 span 정리
#     if cur_type is not None:
#         close(len(tokens))

#     return spans