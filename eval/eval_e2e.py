#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End 답변 정확도 평가

검색 평가(eval_retrieval.py)가 '정답 문서를 찾아왔는가'를 봤다면,
이 스크립트는 '그 문서를 읽고 제대로 답했는가'까지 측정한다.

  질문 -> LLM tool calling -> 도구 실행 -> 최종 답변 -> LLM-as-a-Judge 채점

도구 구현은 DB(tool 테이블)에 등록된 코드를 그대로 exec 하므로
실서비스와 동일한 검색 파이프라인을 탄다.

실행:
  docker exec org-openwebui python ./data/eval_e2e.py --limit 40
"""
import os, re, sys, json, time, asyncio, argparse, sqlite3
import urllib.request
from collections import defaultdict

sys.path.insert(0, ".")
os.environ.setdefault("CORPUS_ROOT", "/app")

VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
DB = "./data/vector_db/webui.db"
MODEL = None
ANS_URL = None
ANS_MODEL = None
JUDGE_URL = None
JUDGE_MODEL = None
FORCE_FIRST_TOOL = False
ANS_THINK_PREFILL = True

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "연구기관의 규정·지침·별표·공지사항을 검색한다. "
                "업무 관련 질문에는 반드시 이 도구를 가장 먼저 호출해야 한다. "
                "내부 지식으로 답하지 말 것."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 질문(문장 형태)"},
                    "top_k": {"type": "integer", "description": "결과 개수(기본 3)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_regulation_file",
            "description": (
                "특정 규정 파일의 전체 내용을 읽는다. "
                "semantic_search 결과만으로 맥락이 부족할 때만 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string",
                                 "description": "semantic_search 결과에서 얻은 파일명"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parent_org_search",
            "description": (
                "상위기관 규정을 검색한다. "
                "semantic_search 로 연구기관 규정을 찾지 못했을 때만 호출한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM = (
    "당신은 연구기관 규정 전문가입니다. 반드시 도구를 사용해 근거를 찾아 답하십시오.\n"
    "- 내부 지식으로 추측하지 마십시오. 규정은 수시로 개정됩니다.\n"
    "- 검색 결과가 부족하면 read_regulation_file 로 원문을 확인하거나 parent_org_search 를 호출하십시오.\n"
    "- 최종 답변은 질문에 대한 사실을 간결하게 서술하십시오. 금액·기간·배점 등 수치는 정확히 인용하십시오."
)

JUDGE_PROMPT = """다음은 사내 규정 질의응답에 대한 채점 과제입니다.

[질문]
{q}

[정답]
{gold}

[응답]
{pred}

[채점 기준]
- 응답이 정답의 핵심 사실(수치, 대상, 조건 등)을 담고 있으면 1점.
- 표현이 달라도 의미가 같으면 정답으로 인정합니다.
- 핵심 수치가 틀리거나, 정답과 다른 내용을 말하거나, 답을 못 찾았다고 하면 0점.
- 정답보다 자세한 내용이 추가된 것은 감점하지 않습니다.

JSON 한 줄로만 출력하십시오. 설명 금지.
{{"score": 0 또는 1, "reason": "20자 이내 사유"}}"""


def _post(payload, timeout=240, retries=3, base=None):
    data = json.dumps(payload).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(
                (base or VLLM) + "/chat/completions", data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == retries - 1:
                return {"__error__": f"{type(e).__name__}: {str(e)[:120]}"}
            time.sleep(2)
    return {"__error__": "unreachable"}


def load_tools():
    """DB에 등록된 도구 코드를 그대로 실행 (실서비스와 동일)"""
    content = sqlite3.connect(DB).execute(
        "SELECT content FROM tool WHERE id='regulation_search'").fetchone()[0]
    ns = {"__name__": "rag_test_e2e"}
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
        return f"[도구 실행 오류: {type(e).__name__}: {str(e)[:100]}]"


async def answer_one(tools, question, max_turns=4):
    """tool calling 루프를 돌려 최종 답변을 얻는다."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    used = []
    for turn in range(max_turns):
        # 첫 턴 도구 호출 강제: 프롬프트 지시가 아닌 API 레벨 강제.
        # SFT로 몸에 밴 '즉답 습관'이 지시문을 무시하는지 검증하기 위한 스위치.
        tc = "required" if (FORCE_FIRST_TOOL and turn == 0) else "auto"
        payload = {
            "model": ANS_MODEL, "messages": msgs, "tools": TOOLS_SPEC,
            "tool_choice": tc, "temperature": 0.1, "max_tokens": 900,
        }
        if ANS_THINK_PREFILL:
            # 기본: 빈 think 블록을 프리필해 사고 생략 (기존 평가와 동일 조건)
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        res = await asyncio.to_thread(lambda: _post(payload, base=ANS_URL))
        if "__error__" in res:
            return f"Error: {res['__error__']}", used
        ch = res["choices"][0]
        msg = ch["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return (msg.get("content") or "").strip(), used

        msgs.append({"role": "assistant", "content": msg.get("content") or "",
                     "tool_calls": tcs})
        for tc in tcs:
            fname = tc["function"]["name"]
            try:
                fargs = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                fargs = {}
            used.append(fname)
            out = await call_tool(tools, fname, fargs)
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                         "name": fname, "content": str(out)[:12000]})
    return "Error: 최대 턴 초과 (도구 호출 반복)", used


async def judge(question, gold, pred):
    if pred.startswith("Error:"):
        return 0, "시스템 오류"
    res = await asyncio.to_thread(lambda: _post({
        "model": JUDGE_MODEL,
        "messages": [{"role": "user",
                      "content": JUDGE_PROMPT.format(q=question, gold=gold, pred=pred[:2500])}],
        "temperature": 0.0, "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120, base=JUDGE_URL))
    if "__error__" in res:
        return 0, "채점 실패"
    txt = (res["choices"][0]["message"].get("content") or "").strip()
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if not m:
        return (1 if re.search(r'"?score"?\s*[:=]\s*1', txt) else 0), "형식오류"
    try:
        j = json.loads(m.group(0))
        return int(j.get("score", 0)), str(j.get("reason", ""))[:40]
    except Exception:
        return 0, "파싱실패"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="./data/rag_test_set_v2.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="./data/eval_e2e_result.json")
    ap.add_argument("--answer-url", default="", help="답변 생성 모델 API (기본: VLLM_URL)")
    ap.add_argument("--answer-model", default="", help="답변 모델명 (기본: 자동)")
    ap.add_argument("--judge-url", default="", help="채점 모델 API (기본: VLLM_URL)")
    ap.add_argument("--force-first-tool", action="store_true",
                    help="첫 턴에 tool_choice=required 로 도구 호출을 API 강제")
    ap.add_argument("--no-think-prefill", action="store_true",
                    help="답변 모델 호출에서 enable_thinking=False 프리필 제거 "
                         "(궤적 SFT 모델은 프리필이 '답변 자리' 신호로 학습되어 도구를 건너뜀)")
    a = ap.parse_args()

    global MODEL, ANS_URL, ANS_MODEL, JUDGE_URL, JUDGE_MODEL

    def first_model(url):
        return json.loads(urllib.request.urlopen(url + "/models", timeout=15)
                          .read())["data"][0]["id"]

    ANS_URL = a.answer_url or VLLM
    ANS_MODEL = a.answer_model or first_model(ANS_URL)
    JUDGE_URL = a.judge_url or VLLM
    JUDGE_MODEL = first_model(JUDGE_URL)
    global FORCE_FIRST_TOOL
    FORCE_FIRST_TOOL = a.force_first_tool
    global ANS_THINK_PREFILL
    ANS_THINK_PREFILL = not a.no_think_prefill
    MODEL = ANS_MODEL
    print(f"답변 모델: {ANS_MODEL} @ {ANS_URL}")
    print(f"채점 모델: {JUDGE_MODEL} @ {JUDGE_URL}")
    print(f"첫 턴 도구 강제: {FORCE_FIRST_TOOL}")

    rows = json.load(open(a.testset, encoding="utf-8"))
    if a.limit:
        # 유형별로 고르게 뽑는다
        byt, picked = defaultdict(list), []
        for r in rows:
            byt[r["type"]].append(r)
        per = max(1, a.limit // max(1, len(byt)))
        for t in sorted(byt):
            picked += byt[t][:per]
        rows = picked[:a.limit]

    tools = load_tools()
    print(f"평가 {len(rows)}건 | 동시 {a.concurrency}")

    sem = asyncio.Semaphore(a.concurrency)
    results, done, t0 = [], [0], time.time()

    async def work(r):
        async with sem:
            pred, used = await answer_one(tools, r["question"])
            score, reason = await judge(r["question"], r["answer"], pred)
            done[0] += 1
            if done[0] % 10 == 0:
                el = time.time() - t0
                print(f"  {done[0]}/{len(rows)}  ({el:.0f}s, "
                      f"평균 {el/done[0]:.1f}s/건)", flush=True)
            return {"type": r["type"], "question": r["question"],
                    "gold": r["answer"], "pred": pred[:1500],
                    "tools_used": used, "score": score, "reason": reason,
                    "source_file": r.get("source_file", ""),
                    "linked_file": r.get("linked_file", "")}

    results = await asyncio.gather(*[work(r) for r in rows])

    # ---------------- 집계 ----------------
    print("\n" + "=" * 60)
    print("End-to-End 답변 정확도")
    print("=" * 60)
    labels = {"A": "규정본문", "B": "별표", "C": "본문+별표 연계", "D": "공지"}
    agg = defaultdict(lambda: [0, 0])
    for r in results:
        agg[r["type"]][0] += r["score"]
        agg[r["type"]][1] += 1
    tot = sum(v[1] for v in agg.values())
    hit = sum(v[0] for v in agg.values())
    for t in ["A", "B", "C", "D"]:
        if agg[t][1]:
            h, n = agg[t]
            print(f"  [{t}] {labels[t]:14s} {100*h/n:5.1f}%  ({h}/{n})")
    if tot:
        print(f"  {'전체':>19s} {100*hit/tot:5.1f}%  ({hit}/{tot})")

    errs = [r for r in results if r["pred"].startswith("Error:")]
    print(f"\n  시스템 오류: {len(errs)}건")
    for e in errs[:3]:
        print(f"    - {e['pred'][:70]}")

    tu = defaultdict(int)
    for r in results:
        for t in set(r["tools_used"]):
            tu[t] += 1
    print(f"\n  도구 사용 (질의당 1회 이상):")
    for k, v in sorted(tu.items(), key=lambda x: -x[1]):
        print(f"    {k:24s} {v}/{len(results)}")
    notool = sum(1 for r in results if not r["tools_used"])
    print(f"    (도구 미사용)            {notool}/{len(results)}")
    print(f"\n  소요 {time.time()-t0:.0f}초")

    json.dump({"total": tot, "hit": hit,
               "by_type": {k: v for k, v in agg.items()},
               "details": results},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  상세 -> {a.out}")


asyncio.run(main())
