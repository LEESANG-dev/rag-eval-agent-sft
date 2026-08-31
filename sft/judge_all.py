# -*- coding: utf-8 -*-
"""35B로 답변 생성(동일 프로토콜) + 3세트 채점"""
import json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = json.loads(urllib.request.urlopen(
    "http://127.0.0.1:8000/v1/models", timeout=15).read())["data"][0]["id"]
SYSTEM = ("당신은 연구기관의 규정 전문가입니다. "
          "제공된 검색 결과만을 근거로 답변하며, 금액·기간·배점 등 수치는 원문 그대로 정확히 인용합니다. "
          "답변 마지막에 근거 문서명을 명시합니다. 검색 결과에 없는 내용은 추측하지 않습니다.")
JUDGE = """다음은 사내 규정 질의응답 채점 과제입니다.
[질문]\n{q}\n[정답]\n{g}\n[응답]\n{p}
[기준] 핵심 사실(수치·대상·조건)이 정답과 일치하면 1, 다르거나 못 찾으면 0. 표현 차이는 무시.
JSON 한 줄만: {{"score": 0 또는 1}}"""

def call(messages, max_tokens, retries=3):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0.0,
                       "max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return (json.loads(r.read().decode())["choices"][0]["message"]
                        .get("content") or "").strip()
        except Exception as e:
            if i == retries - 1:
                return f"Error: {type(e).__name__}"
            time.sleep(3)

# 1) 35B 답변 생성 (gold context, 동일 프롬프트)
ctx = json.load(open("/tmp/eval_ctx.json", encoding="utf-8"))
t0 = time.time()
def gen(r):
    u = (f"다음 검색 결과를 근거로 질문에 답하십시오.\n\n[검색 결과]\n{r['context']}"
         f"\n\n[질문]\n{r['question']}")
    return call([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": u}], 600)
with ThreadPoolExecutor(max_workers=5) as ex:
    preds = list(ex.map(gen, ctx))
out = [{**{k: r[k] for k in ("type", "question", "gold", "source_file")}, "pred": p}
       for r, p in zip(ctx, preds)]
json.dump(out, open("/tmp/answers_qwen35b.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"35B 답변 생성 {len(out)}건 {time.time()-t0:.0f}s")

# 2) 3세트 채점
def judge_one(r):
    if r["pred"].startswith("Error"):
        return 0
    txt = call([{"role": "user", "content":
                 JUDGE.format(q=r["question"], g=r["gold"], p=r["pred"][:2200])}], 120)
    m = re.search(r'"score"\s*:\s*(\d)', txt or "")
    return int(m.group(1)) if m else 0

for name in ("qwen3-8b", "org-lora", "qwen35b"):
    rows = json.load(open(f"/tmp/answers_{name}.json", encoding="utf-8"))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        scores = list(ex.map(judge_one, rows))
    agg = defaultdict(lambda: [0, 0])
    for r, s in zip(rows, scores):
        r["score"] = s
        agg[r["type"]][0] += s
        agg[r["type"]][1] += 1
    json.dump(rows, open(f"/tmp/judged_{name}.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    tot = sum(v[1] for v in agg.values()); hit = sum(v[0] for v in agg.values())
    parts = " / ".join(f"{t}:{100*agg[t][0]/agg[t][1]:.1f}%" for t in "ABCD" if agg[t][1])
    print(f"[{name}] 전체 {100*hit/tot:.1f}% ({hit}/{tot})  {parts}  ({time.time()-t0:.0f}s)")
