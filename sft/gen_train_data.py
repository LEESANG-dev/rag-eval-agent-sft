#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 학습 데이터 생성기 — 35B 증류 (연구기관 규정 QA)

평가셋 생성기(gen_rag_testset.py)의 품질 필터를 재사용하되, 학습용으로 3가지를 추가:
  1) 근거 검증: 정답에 포함된 수치가 컨텍스트에 실제로 존재해야 함 (환각 차단)
  2) 오염 방지: 평가셋 394건과 질문이 겹치면 폐기 (문자 3-gram 유사도)
  3) 포맷: 실서비스 RAG와 동일한 (검색결과 + 질문 -> 근거 인용 답변) chat 형식

출력: JSONL — {"messages":[system,user,assistant], "type","source_file"}

사용:
  python3 gen_train_data.py --pilot            # 유형별 6건
  python3 gen_train_data.py --full             # A1200 B1200 C600 D200
"""
import os, re, json, glob, random, argparse, sys, time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("CORPUS_ROOT", "/data/regulations")
VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = ""
EVAL_SET = os.path.join(ROOT, "scripts", "rag_test_set_v2.json")

DIR_REG = os.path.join(ROOT, "regulation_txt_merged")
DIR_APP = os.path.join(ROOT, "regulation_md_merged")
DIR_NOT = os.path.join(ROOT, "crawled_data", "analyzed")

random.seed(20260827)

PLAN_FULL = {"A": 1200, "B": 1200, "C": 600, "D": 200}
PLAN_PILOT = {"A": 6, "B": 6, "C": 6, "D": 6}
PER_CALL = 4          # 호출당 QA 수

SYSTEM_PROMPT = (
    "당신은 연구기관의 규정 전문가입니다. "
    "제공된 검색 결과만을 근거로 답변하며, 금액·기간·배점 등 수치는 원문 그대로 정확히 인용합니다. "
    "답변 마지막에 근거 문서명을 명시합니다. 검색 결과에 없는 내용은 추측하지 않습니다."
)

GEN_RULES = """[출제 금지]
1. "몇 번째 항목" 등 순번 암기형 질문 금지.
2. "이 문서에서", "위 내용에 따르면" 등 문서 지시 표현 금지 — 질문만 읽어도 자립적이어야 한다.
3. 정답이 질문에 노출되는 문항 금지.
4. 문서에 근거 없는 내용 금지. 정답의 모든 수치는 반드시 문서에 있는 값이어야 한다.
5. "없습니다/명시되어 있지 않습니다"로 끝나는 정답 금지.

[답변 작성 방침]
- 실무자의 질문에 답하듯 완결된 문장으로 작성한다. 질문 20자 이상, 답변 30자 이상.
- 수치(금액/기간/비율/배점/인원)를 묻는 질문을 우선하고, 답변에 그 값을 그대로 인용한다.
- 답변 마지막 줄에 "근거: {문서명}" 을 붙인다.

[출력 형식] JSON 배열만. 설명·코드펜스 금지.
[{"question": "...", "answer": "..."}]
"""

PROMPTS = {
    "A": "다음은 연구기관 내부 규정 문서다. 이 문서만 근거로 질의응답 {n}개를 만들어라.\n\n"
         + GEN_RULES + "\n[문서: {name}]\n{body}\n",
    "B": "다음은 연구기관 규정의 '별표'(기준표) 문서다. 표 안의 구체적 값(금액/등급/기간/배점)을 묻는 "
         "질의응답 {n}개를 만들어라. 특히 **표의 행과 열을 정확히 교차 참조해야 답할 수 있는** 질문을 우선하라. "
         "(예: 'X등급의 Y항목 기준은?') 질문에 '별표'라는 단어는 쓰지 마라.\n\n"
         + GEN_RULES + "\n[별표: {name}]\n{body}\n",
    "C": "아래에 연구기관 규정의 본문 조문과 그 조문이 참조하는 별표가 함께 있다. "
         "**본문의 조건·대상과 별표의 구체적 수치를 모두 알아야 답할 수 있는** 질의응답 {n}개를 만들어라. "
         "정답에는 별표의 수치가 반드시 그대로 인용되어야 한다. 질문에 '별표'라는 단어는 쓰지 마라.\n\n"
         + GEN_RULES + "\n[본문: {reg_name}]\n{reg_body}\n\n[별표: {app_name}]\n{app_body}\n",
    "D": "다음은 연구기관 내부 공지사항이다. 일정·대상·신청 방법·담당 부서 등 공지에서 확인 가능한 사실을 묻는 "
         "질의응답 {n}개를 만들어라.\n\n" + GEN_RULES + "\n[공지: {name}]\n{body}\n",
}

# ---------------- 유틸 (gen_rag_testset.py 재사용) ----------------
def fill(tpl, **kw):
    for k, v in kw.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def read_text(path, limit=9000):
    for enc in ("utf-8", "cp949"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()[:limit]
        except UnicodeDecodeError:
            continue
        except Exception:
            return ""
    return ""


def strip_md_header(t):
    return re.sub(r"^#.*?\n(-\s*\*\*.*?\n)*\s*---\s*\n", "", t, flags=re.DOTALL)


def call_llm(prompt, max_tokens=1600, temperature=0.7, retries=3):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    data = json.dumps(body).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(VLLM, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                j = json.loads(r.read().decode())
            return (j["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            if i == retries - 1:
                return ""
            time.sleep(2)
    return ""


def parse_qa(raw):
    if not raw:
        return []
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in arr if isinstance(arr, list) else []:
        if isinstance(it, dict):
            q = str(it.get("question", "")).strip()
            a = str(it.get("answer", "")).strip()
            if q and a:
                out.append({"question": q, "answer": a})
    return out


BAD_Q = re.compile(r"몇\s*번\s*째|몇\s*번째|이\s*문서|본\s*문서|위\s*내용|해당\s*문서|별표")
BAD_A = re.compile(r"없습니다|명시되어\s*있지|해당\s*없")
NUM = re.compile(r"\d[\d,.]*")

FORM_HINT = re.compile(r"별지|서\s*식|양\s*식|신청서|보고서|확인서|계획서|대장|증명서|동의서|위임장|이력서|서약서")


def table_score(text):
    rows = len(re.findall(r"^\s*\|.*\|\s*$", text, flags=re.M))
    nums = len(re.findall(r"\d", text))
    return rows, nums


def is_usable_appendix(path, text):
    rows, nums = table_score(text)
    if len(text) < 500 or nums < 25:
        return False
    if rows < 4 and nums < 60:
        return False
    if FORM_HINT.search(os.path.basename(path)) and rows < 8:
        return False
    return True


# ---------------- 오염 방지: 평가셋 질문과 중복 차단 ----------------
def trigrams(s):
    s = re.sub(r"\s+", "", s)
    return {s[i:i+3] for i in range(max(1, len(s) - 2))}


class EvalGuard:
    def __init__(self, path):
        self.qs = []
        try:
            for r in json.load(open(path, encoding="utf-8")):
                q = str(r.get("question", ""))
                self.qs.append((q, trigrams(q)))
            print(f"오염 가드: 평가셋 질문 {len(self.qs)}건 로드")
        except Exception as e:
            print(f"경고: 평가셋 로드 실패({e}) — 중복 필터 없이 진행")

    def is_dup(self, q, thresh=0.55):
        tg = trigrams(q)
        for _, etg in self.qs:
            inter = len(tg & etg)
            if inter and inter / max(1, min(len(tg), len(etg))) >= thresh:
                return True
        return False


def quality_ok(qa, context, kind, guard):
    q, a = qa["question"], qa["answer"]
    if len(q) < 20 or len(a) < 30:
        return False, "길이"
    if BAD_Q.search(q):
        return False, "금지패턴"
    if BAD_A.search(a):
        return False, "부정형"
    if len(a) > 4 and a in q:
        return False, "정답노출"
    # 근거 검증: 답변의 수치가 컨텍스트에 존재해야 함 (콤마 제거 후 비교)
    ctx_flat = context.replace(",", "")
    nums = NUM.findall(a.replace(",", ""))
    nums = [n for n in nums if len(n) >= 2]          # 한 자리 수는 노이즈
    if kind in ("B", "C") and not nums:
        return False, "수치없음"
    for n in nums:
        if n not in ctx_flat:
            return False, "근거불일치"
    if guard.is_dup(q):
        return False, "평가셋중복"
    return True, ""


def to_sample(kind, src_name, context, qa):
    user = (f"다음 검색 결과를 근거로 질문에 답하십시오.\n\n"
            f"[검색 결과]\n{context}\n\n[질문]\n{qa['question']}")
    ans = qa["answer"]
    if "근거:" not in ans:
        ans = ans.rstrip() + f"\n\n근거: {src_name}"
    return {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": ans}],
            "type": kind, "source_file": src_name}


# ---------------- 소스 수집 ----------------
def collect():
    regs = [p for p in sorted(glob.glob(os.path.join(DIR_REG, "*.txt")))
            if os.path.getsize(p) > 800]
    apps = []
    for p in sorted(glob.glob(os.path.join(DIR_APP, "*.md"))):
        t = strip_md_header(read_text(p, 20000))
        if is_usable_appendix(p, t):
            apps.append(p)
    nots = [p for p in sorted(glob.glob(os.path.join(DIR_NOT, "*")))
            if os.path.isfile(p) and os.path.getsize(p) > 500]

    reg_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in regs}
    pairs = []
    for ap in apps:
        m = re.match(r"^(.*)_별표\s*제?[\d\-]+\s*호?$",
                     os.path.splitext(os.path.basename(ap))[0])
        if m and m.group(1) in reg_by_stem:
            pairs.append((reg_by_stem[m.group(1)], ap))
    return regs, apps, nots, pairs


def find_ref(reg_text, app_name):
    m = re.search(r"별표\s*제?\s*([\d\-]+)\s*호?", app_name)
    if not m:
        return reg_text[:3500]
    hit = re.search(r"별표\s*제?\s*" + re.escape(m.group(1)) + r"\s*호?", reg_text)
    if not hit:
        return reg_text[:3500]
    s = max(0, hit.start() - 1500)
    return reg_text[s:hit.start() + 1500]


# ---------------- 생성 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()

    global MODEL
    MODEL = json.loads(urllib.request.urlopen(
        VLLM.replace("/chat/completions", "/models"), timeout=15).read())["data"][0]["id"]
    print("모델:", MODEL)

    plan = PLAN_PILOT if a.pilot else PLAN_FULL
    out = a.out or ("/tmp/train_pilot.jsonl" if a.pilot
                    else os.path.join(ROOT, "scripts", "sft_train_data.jsonl"))

    guard = EvalGuard(EVAL_SET)
    regs, apps, nots, pairs = collect()
    print(f"소스: 규정 {len(regs)} / 별표 {len(apps)} / 공지 {len(nots)} / 쌍 {len(pairs)}")
    random.shuffle(regs); random.shuffle(apps); random.shuffle(nots); random.shuffle(pairs)

    stats = defaultdict(int)
    samples = []
    t0 = time.time()

    def job(kind, item):
        """(prompt, context, src_name) 만들고 호출-필터-샘플화까지"""
        if kind == "A":
            name = os.path.basename(item)
            body = read_text(item)
            prompt = fill(PROMPTS["A"], n=PER_CALL, name=name, body=body)
            ctx, src = body, name
        elif kind == "B":
            name = os.path.basename(item)
            body = strip_md_header(read_text(item))
            prompt = fill(PROMPTS["B"], n=PER_CALL, name=name, body=body)
            ctx, src = body, name
        elif kind == "C":
            rp, appp = item
            rtxt = read_text(rp, 20000)
            atxt = strip_md_header(read_text(appp, 7000))
            excerpt = find_ref(rtxt, os.path.basename(appp))
            prompt = fill(PROMPTS["C"], n=PER_CALL,
                          reg_name=os.path.basename(rp), reg_body=excerpt,
                          app_name=os.path.basename(appp), app_body=atxt)
            ctx = excerpt + "\n\n" + atxt
            src = os.path.basename(rp) + " + " + os.path.basename(appp)
        else:
            name = os.path.basename(item)
            body = read_text(item, 6000)
            prompt = fill(PROMPTS["D"], n=PER_CALL, name=name, body=body)
            ctx, src = body, name

        got = []
        for qa in parse_qa(call_llm(prompt)):
            ok, why = quality_ok(qa, ctx, kind, guard)
            stats[f"{kind}:{'ok' if ok else why}"] += 1
            if ok:
                got.append(to_sample(kind, src, ctx, qa))
        return got

    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for kind, pool in [("A", regs), ("B", apps), ("C", pairs), ("D", nots)]:
            need = plan.get(kind, 0)
            if not need:
                continue
            got = 0
            i = 0
            # 문서를 순환하며 목표 수량까지 (문서당 여러 번 호출될 수 있음 — temperature 로 다양화)
            batch = 12
            while got < need and i < len(pool) * 3:
                items = [pool[(i + k) % len(pool)] for k in range(batch)]
                i += batch
                for res in ex.map(lambda it: job(kind, it), items):
                    samples.extend(res)
                    got += len(res)
                    if got >= need:
                        break
                print(f"  [{kind}] {min(got, need)}/{need}  ({time.time()-t0:.0f}s)", flush=True)
            stats[f"{kind}:수집"] = min(got, need)

    # 문서 단위 train/val 분리 (같은 문서가 양쪽에 걸치지 않게)
    by_doc = defaultdict(list)
    for s in samples:
        by_doc[s["source_file"]].append(s)
    docs = list(by_doc)
    random.shuffle(docs)
    n_val_docs = max(1, len(docs) // 20)
    val_docs = set(docs[:n_val_docs])
    train = [s for d in docs[n_val_docs:] for s in by_doc[d]]
    val = [s for d in val_docs for s in by_doc[d]]
    random.shuffle(train)

    with open(out, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    vout = out.replace(".jsonl", "_val.jsonl")
    with open(vout, "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\ntrain {len(train)}건 -> {out}")
    print(f"val   {len(val)}건 ({n_val_docs}개 문서) -> {vout}")
    print("\n[필터 통계]")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"\n소요 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
