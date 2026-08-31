# -*- coding: utf-8 -*-
"""base/LoRA 8B로 평가셋 답변 생성 (학습과 동일한 프롬프트 형식)"""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/v1/chat/completions"
SYSTEM = ("당신은 연구기관의 규정 전문가입니다. "
          "제공된 검색 결과만을 근거로 답변하며, 금액·기간·배점 등 수치는 원문 그대로 정확히 인용합니다. "
          "답변 마지막에 근거 문서명을 명시합니다. 검색 결과에 없는 내용은 추측하지 않습니다.")

rows = json.load(open("./data/eval_ctx.json", encoding="utf-8"))
print("평가", len(rows), "건 x 2모델")

def ask(model, r, retries=3):
    user = (f"다음 검색 결과를 근거로 질문에 답하십시오.\n\n[검색 결과]\n{r['context']}"
            f"\n\n[질문]\n{r['question']}")
    body = json.dumps({"model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": 600,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                j = json.loads(resp.read().decode())
            return (j["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            if i == retries - 1:
                return f"Error: {type(e).__name__}"
            time.sleep(3)

for model in ("qwen3-8b", "org-lora"):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        preds = list(ex.map(lambda r: ask(model, r), rows))
    out = [{**{k: r[k] for k in ("type", "question", "gold", "source_file")},
            "pred": p} for r, p in zip(rows, preds)]
    path = f"./data/answers_{model}.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    errs = sum(1 for p in preds if p.startswith("Error"))
    print(f"{model}: {len(out)}건 {time.time()-t0:.0f}s (오류 {errs}) -> {path}")
