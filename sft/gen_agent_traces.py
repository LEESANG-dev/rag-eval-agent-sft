#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2차 학습 데이터: 35B 에이전트 궤적 증류

1차 데이터는 '컨텍스트 제공 -> 즉답' 형식이라 도구 호출 행동이 학습에서 지워졌다
(E2E 67.3%, 도구 미호출 77건). 이번에는 35B가 실제 파이프라인에서 일하는
전 과정(질문 -> semantic_search 호출 -> 도구 결과 -> 답변)을 통째로 녹화한다.

품질 필터:
  - 35B의 최종 답변이 검증된 정답과 일치(judge)할 때만 채택 — 성공한 행동만 증류
  - 도구를 한 번도 안 부른 궤적은 폐기 — '즉답 습관'이 다시 들어가는 것을 차단

실행: docker exec org-openwebui python ./data/gen_agent_traces.py \
        --src ./data/sft_train_data.jsonl --out ./data/agent_traces.jsonl
"""
import os, re, sys, json, time, argparse, asyncio, sqlite3, urllib.request

sys.path.insert(0, ".")
os.environ.setdefault("CORPUS_ROOT", "/app")

VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
MODEL = None

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "semantic_search",
        "description": ("연구기관의 규정·지침·별표·공지사항을 검색한다. "
                        "업무 관련 질문에는 반드시 이 도구를 가장 먼저 호출해야 한다. "
                        "내부 지식으로 답하지 말 것."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "검색할 질문(문장 형태)"},
            "top_k": {"type": "integer", "description": "결과 개수(기본 3)"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_regulation_file",
        "description": ("특정 규정 파일의 전체 내용을 읽는다. "
                        "semantic_search 결과만으로 맥락이 부족할 때만 사용한다."),
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "검색 결과의 문서 ID 또는 파일명"}},
            "required": ["filename"]}}},
    {"type": "function", "function": {
        "name": "parent_org_search",
        "description": "상위기관 규정을 검색한다. 연구기관 규정에서 못 찾았을 때만 호출한다.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]

SYSTEM = (
    "당신은 연구기관 규정 전문가입니다. 반드시 도구를 사용해 근거를 찾아 답하십시오.\n"
    "- 내부 지식으로 추측하지 마십시오. 규정은 수시로 개정됩니다.\n"
    "- 검색 결과가 부족하면 read_regulation_file 로 원문을 확인하거나 parent_org_search 를 호출하십시오.\n"
    "- 해당하는 항목이 여러 개면 하나도 빠짐없이 모두 열거하십시오.\n"
    "- 금액·기간·배점 등 수치는 원문 그대로 정확히 인용하고, 답변 끝에 근거 문서를 명시하십시오."
)

JUDGE_PROMPT = """다음은 사내 규정 질의응답 채점 과제입니다.

[질문]
{q}

[정답]
{gold}

[응답]
{pred}

[기준] 응답이 정답의 핵심 사실(수치·대상·조건)을 담으면 1, 아니면 0.
표현이 달라도 의미가 같으면 1. 더 자세한 것은 감점하지 않음.
JSON 한 줄만: {{"score": 0 또는 1}}"""


def _post(payload, timeout=240, retries=3):
    data = json.dumps(payload).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(VLLM + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == retries - 1:
                return {"__error__": f"{type(e).__name__}"}
            time.sleep(2)


def load_tools():
    content = sqlite3.connect("./data/vector_db/webui.db").execute(
        "SELECT content FROM tool WHERE id='regulation_search'").fetchone()[0]
    ns = {"__name__": "rag_trace"}
    exec(compile(content, "<regulation_search:db>", "exec"), ns)
    return ns["Tools"]()


async def call_tool(tools, name, args):
    try:
        fn = getattr(tools, name, None)
        if fn is None:
            return f"[도구 없음: {name}]"
        if asyncio.iscoroutinefunction(fn):
            return await fn(**args)
        return fn(**args)
    except Exception as e:
        return f"[도구 실행 오류: {type(e).__name__}]"


async def run_agent(tools, question, max_turns=4):
    """35B 에이전트 루프를 돌리고 전체 메시지 궤적을 반환"""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    n_tool = 0
    for _ in range(max_turns):
        res = await asyncio.to_thread(_post, {
            "model": MODEL, "messages": msgs, "tools": TOOLS_SPEC,
            "tool_choice": "auto", "temperature": 0.2, "max_tokens": 900,
            "chat_template_kwargs": {"enable_thinking": False}})
        if "__error__" in res:
            return None, n_tool, "api오류"
        msg = res["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            final = (msg.get("content") or "").strip()
            if not final:
                return None, n_tool, "빈응답"
            msgs.append({"role": "assistant", "content": final})
            return msgs, n_tool, ""
        # 도구 호출 턴 기록 (API가 준 형식 그대로 보존)
        msgs.append({"role": "assistant", "content": msg.get("content") or "",
                     "tool_calls": tcs})
        for tc in tcs:
            n_tool += 1
            try:
                fargs = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                fargs = {}
            out = await call_tool(tools, tc["function"]["name"], fargs)
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                         "name": tc["function"]["name"],
                         "content": str(out)[:12000]})
    return None, n_tool, "턴초과"


async def judge_ok(question, gold, pred):
    res = await asyncio.to_thread(_post, {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": JUDGE_PROMPT.format(q=question, gold=gold, pred=pred[:2500])}],
        "temperature": 0.0, "max_tokens": 100,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
    if "__error__" in res:
        return False
    txt = (res["choices"][0]["message"].get("content") or "")
    return bool(re.search(r'"score"\s*:\s*1', txt))


Q_RE = re.compile(r"\[질문\]\s*(.+?)\s*$", re.DOTALL)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()

    global MODEL
    MODEL = json.loads(urllib.request.urlopen(VLLM + "/models", timeout=15)
                       .read())["data"][0]["id"]
    print("선생 모델:", MODEL)

    rows = []
    for line in open(a.src, encoding="utf-8"):
        r = json.loads(line)
        user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        gold = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
        m = Q_RE.search(user)
        if not m or not gold:
            continue
        rows.append({"type": r.get("type", "?"), "source_file": r.get("source_file", ""),
                     "question": m.group(1).strip(), "gold": gold.strip()})
    if a.limit:
        rows = rows[:a.limit]
    print(f"원본 {len(rows)}건 처리 시작 (동시 {a.concurrency})")

    tools = load_tools()
    sem = asyncio.Semaphore(a.concurrency)
    stats = {"ok": 0, "judge탈락": 0, "도구미사용": 0, "api오류": 0, "빈응답": 0, "턴초과": 0}
    out_rows, done, t0 = [], [0], time.time()

    async def work(r):
        async with sem:
            traj, n_tool, err = await run_agent(tools, r["question"])
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  {done[0]}/{len(rows)} ({time.time()-t0:.0f}s) 채택 {stats['ok']}", flush=True)
            if traj is None:
                stats[err] = stats.get(err, 0) + 1
                return
            if n_tool == 0:
                stats["도구미사용"] += 1        # 즉답 궤적은 학습에 넣지 않는다
                return
            final = traj[-1]["content"]
            if not await judge_ok(r["question"], r["gold"], final):
                stats["judge탈락"] += 1
                return
            stats["ok"] += 1
            out_rows.append({"type": r["type"], "source_file": r["source_file"],
                             "messages": traj, "tools": TOOLS_SPEC})

    await asyncio.gather(*[work(r) for r in rows])

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n채택 {stats['ok']}/{len(rows)} -> {a.out}")
    print("필터 통계:", {k: v for k, v in stats.items() if v})
    from collections import Counter
    print("유형 분포:", dict(Counter(r["type"] for r in out_rows)))
    nt = Counter(sum(1 for m in r["messages"] if m["role"] == "tool") for r in out_rows)
    print("궤적당 도구 호출 수:", dict(sorted(nt.items())))
    print(f"소요 {time.time()-t0:.0f}초")


asyncio.run(main())
