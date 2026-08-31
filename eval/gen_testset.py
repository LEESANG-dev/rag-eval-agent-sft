#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연구기관 RAG 테스트셋 생성기 (v2)

기존 셋(rag_test_set.json)의 결함을 교정한 재설계:
  - 별표 0% -> B/C 유형으로 커버리지 확보
  - 순번 암기형/문서 지시형 문항 프롬프트 단계에서 차단
  - 본문<->별표 연계형(C) 추가: GraphRAG/별표 동반조회 효과 측정용

사용:
  python3 gen_testset.py --pilot          # 유형별 5건 파일럿
  python3 gen_testset.py --full           # 전체 400건
"""
import os, re, json, glob, random, argparse, urllib.request, urllib.error, sys
from collections import defaultdict

ROOT = os.environ.get("CORPUS_ROOT", "/data/regulations")
VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "")

DIR_REG = os.path.join(ROOT, "regulation_txt_merged")
DIR_APP = os.path.join(ROOT, "regulation_md_merged")
DIR_NOT = os.path.join(ROOT, "crawled_data", "analyzed")

random.seed(20260826)

PLAN_FULL = {"A": 140, "B": 140, "C": 80, "D": 40}
PLAN_PILOT = {"A": 5, "B": 5, "C": 5, "D": 5}

# ---------------------------------------------------------------- 공통 규칙
RULES = """[출제 금지 - 반드시 지킬 것]
1. "몇 번째 항목", "몇 번 째", "순서상 어디" 처럼 순번을 세는 질문 금지.
2. "이 문서에서", "위 내용에 따르면", "본 지침의" 처럼 문서를 가리키는 표현 금지.
   -> 질문만 읽어도 무엇을 묻는지 알 수 있어야 한다.
3. 정답이 질문 안에 그대로 노출되는 질문 금지.
4. 문서에 근거가 없는 내용을 지어내지 말 것.
5. 정답이 "없습니다", "명시되어 있지 않습니다" 로 끝나는 문항 금지.

[출제 방침]
- 실제 연구소 직원이 업무 중 물어볼 법한 자연스러운 질문으로 작성한다.
- 금액, 기간, 비율, 자격요건, 절차처럼 근거가 분명한 사실을 묻는다.
- 질문은 20자 이상, 정답은 15자 이상으로 구체적으로 작성한다.
- 정답은 문서 원문의 표현을 근거로 하되, 완결된 문장으로 쓴다.

[출력 형식]
JSON 배열만 출력한다. 설명, 코드펜스, 생각 과정을 절대 붙이지 않는다.
[{"question": "...", "answer": "..."}]
"""

PROMPTS = {
    "A": """다음은 연구기관의 내부 규정 문서다.
이 문서의 내용만 근거로 질의응답 {n}개를 만들어라.

""" + RULES + """
[문서: {name}]
{body}
""",
    "B": """다음은 연구기관 규정의 '별표'(기준표/서식) 문서다. 표 형태의 수치·기준이 담겨 있다.
이 별표의 내용만 근거로 질의응답 {n}개를 만들어라.

[별표 출제 지침]
- 표 안의 구체적인 값(금액, 등급, 기간, 대상, 배점 등)을 묻는 질문을 우선한다.
- "별표 제N호에 따르면" 같은 표현은 쓰지 말고, 실무자가 묻듯 자연스럽게 작성한다.
  (예: "부장급 국내출장 일비는 얼마인가요?")

""" + RULES + """
[별표: {name}]
{body}
""",
    "C": """아래에는 연구기관 규정의 '본문 조문'과 그 조문이 참조하는 '별표'가 함께 주어진다.
**본문과 별표를 모두 봐야만 답할 수 있는** 질의응답 {n}개를 만들어라.

[연계형 출제 지침 - 가장 중요]
- **정답에는 별표 표 안의 구체적인 값(금액/기간/비율/등급/인원 등)을 반드시 그대로 인용해야 한다.**
  별표의 숫자가 정답에 들어가지 않으면 그 문항은 실패한 문항이다.
- 본문은 "누구에게/어떤 경우에 적용되는지"를, 별표는 "얼마인지"를 제공한다.
  두 정보를 합쳐야 완성되는 질문을 만들어라.
- 본문만 읽고도 답할 수 있는 질문(제출 시기, 결재 라인, 담당 부서 등)은 절대 만들지 말 것.
- 질문에 "별표"라는 단어를 쓰지 말 것. 실무자는 별표 번호를 모른 채 묻는다.

[좋은 예]
  Q: 3급 직원이 국내로 이전할 때 받을 수 있는 이전비 지급 한도는 얼마인가요?
  A: 3급 직원의 국내 이전비는 최대 1,200,000원까지 지급된다.   <- 별표의 금액이 정답에 포함됨
[나쁜 예]
  Q: 이전비를 신청하려면 어디에 제출해야 하나요?
  A: 총무부에 제출한다.                                        <- 본문만으로 답 가능. 금지

""" + RULES + """
[본문: {reg_name}]
{reg_body}

[별표: {app_name}]
{app_body}
""",
    "D": """다음은 연구기관 내부 공지사항 문서다.
이 공지의 내용만 근거로 질의응답 {n}개를 만들어라.

[공지 출제 지침]
- 일정, 대상, 신청 방법, 담당 부서처럼 공지에서 실제로 확인할 수 있는 사실을 묻는다.

""" + RULES + """
[공지: {name}]
{body}
""",
}


# ---------------------------------------------------------------- 유틸
def fill(tpl, **kw):
    """str.format 대신 사용 - 프롬프트에 JSON 중괄호가 들어있어 format은 못 씀"""
    for k, v in kw.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def read_text(path, limit=12000):
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
    """변환기가 붙인 메타 헤더 제거"""
    return re.sub(r"^#.*?\n(-\s*\*\*.*?\n)*\s*---\s*\n", "", t, flags=re.DOTALL)


def call_llm(prompt, max_tokens=2000, temperature=0.4, retries=3):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(VLLM, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                j = json.loads(r.read().decode())
            msg = j["choices"][0]["message"]
            return (msg.get("content") or "").strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"    [LLM 실패] {type(e).__name__}: {e}", file=sys.stderr)
                return ""
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
        if not isinstance(it, dict):
            continue
        q = str(it.get("question", "")).strip()
        a = str(it.get("answer", "")).strip()
        if q and a:
            out.append({"question": q, "answer": a})
    return out


BAD_Q = re.compile(r"몇\s*번\s*째|몇\s*번째|이\s*문서|본\s*문서|위\s*내용|해당\s*문서|본\s*지침에서|이\s*지침에서")
BAD_A = re.compile(r"없습니다|명시되어\s*있지|해당\s*없")

# 서식/양식 판별: 빈칸 서식은 QA 소재가 없다
FORM_HINT = re.compile(r"별지|서\s*식|양\s*식|신청서|보고서|확인서|계획서|대장|증명서|동의서|위임장|이력서|서약서")


def table_score(text):
    """표 형태 기준표인지 점수화. (표 행 수, 숫자 개수)"""
    rows = len(re.findall(r"^\s*\|.*\|\s*$", text, flags=re.M))
    nums = len(re.findall(r"\d", text))
    return rows, nums


def is_usable_appendix(path, text):
    """수치·기준이 담긴 별표만 통과 (빈칸 서식 배제)"""
    rows, nums = table_score(text)
    if len(text) < 500:
        return False
    if nums < 25:                      # 숫자가 거의 없으면 기준표가 아님
        return False
    if rows < 4 and nums < 60:         # 표도 아니고 숫자도 적으면 제외
        return False
    name = os.path.basename(path)
    if FORM_HINT.search(name) and rows < 8:
        return False
    return True


def appendix_only_values(app_text, reg_text):
    """별표에만 있고 본문에는 없는 '값'(숫자/금액/기간) 추출"""
    cand = set(re.findall(r"\d[\d,]*\s*(?:원|만원|천원|일|개월|년|시간|%|점|급|호|명|배)", app_text))
    cand |= set(re.findall(r"\d[\d,]{2,}", app_text))
    return {c.strip() for c in cand if c.strip() and c.strip() not in reg_text}


def c_type_is_linked(qa, app_only):
    """정답에 별표에만 있는 값이 실제로 들어갔는지 검증"""
    a = qa["answer"]
    return any(v in a for v in app_only)


def quality_ok(qa):
    q, a = qa["question"], qa["answer"]
    if len(q) < 20 or len(a) < 15:
        return False, "길이 미달"
    if BAD_Q.search(q):
        return False, "금지 패턴(질문)"
    if BAD_A.search(a):
        return False, "부정형 정답"
    if len(a) > 4 and a in q:
        return False, "정답 노출"
    if "별표" in q:
        return False, "질문에 별표 노출"
    return True, ""


# ---------------------------------------------------------------- 소스 수집
def collect_sources():
    regs = sorted(glob.glob(os.path.join(DIR_REG, "*.txt")))
    apps = sorted(glob.glob(os.path.join(DIR_APP, "*.md")))
    nots = sorted(glob.glob(os.path.join(DIR_NOT, "*")))
    nots = [p for p in nots if os.path.isfile(p)]

    apps = [p for p in apps if os.path.getsize(p) > 400]
    regs = [p for p in regs if os.path.getsize(p) > 800]
    nots = [p for p in nots if os.path.getsize(p) > 500]

    # 별표는 '수치·기준표'만 남긴다 (빈칸 서식 제외)
    before = len(apps)
    apps = [p for p in apps if is_usable_appendix(p, strip_md_header(read_text(p, 20000)))]
    print(f"별표 필터: {before} -> {len(apps)} (서식/빈약 제외 {before-len(apps)})")

    # 본문 <-> 별표 짝짓기: "규정명_별표 제N호.md" -> "규정명.txt"
    reg_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in regs}
    pairs = []
    for ap in apps:
        base = os.path.splitext(os.path.basename(ap))[0]
        m = re.match(r"^(.*)_별표\s*제?[\d\-]+\s*호?$", base)
        if not m:
            continue
        rp = reg_by_stem.get(m.group(1))
        if rp:
            pairs.append((rp, ap))
    return regs, apps, nots, pairs


def find_ref_article(reg_text, app_name):
    """별표 번호를 참조하는 조문 구간을 뽑아낸다."""
    m = re.search(r"별표\s*제?\s*([\d\-]+)\s*호?", app_name)
    if not m:
        return reg_text[:4000]
    num = m.group(1)
    pat = re.compile(r"별표\s*제?\s*" + re.escape(num) + r"\s*호?")
    hit = pat.search(reg_text)
    if not hit:
        return reg_text[:4000]
    start = max(0, hit.start() - 1500)
    return reg_text[start:hit.start() + 1500]


# ---------------------------------------------------------------- 생성
def generate(plan, per_doc=2):
    regs, apps, nots, pairs = collect_sources()
    print(f"소스: 규정 {len(regs)} / 별표 {len(apps)} / 공지 {len(nots)} / 본문-별표쌍 {len(pairs)}")

    random.shuffle(regs); random.shuffle(apps); random.shuffle(nots); random.shuffle(pairs)
    rows, stats = [], defaultdict(int)

    def emit(kind, need, iterator, build):
        got = 0
        for item in iterator:
            if got >= need:
                break
            info = build(item)
            if not info:
                continue
            prompt, meta = info[0], info[1]
            check = info[2] if len(info) > 2 else None
            n = min(per_doc, need - got)
            qas = parse_qa(call_llm(prompt.replace("{n}", str(n))))
            for qa in qas[:n]:
                ok, why = quality_ok(qa)
                if ok and check is not None and not check(qa):
                    ok, why = False, "연계 미충족(별표값 없음)"
                stats[f"{kind}:{'ok' if ok else why}"] += 1
                if not ok:
                    continue
                rows.append({"type": kind, **meta, **qa})
                got += 1
                if got >= need:
                    break
            print(f"  [{kind}] {got}/{need}", end="\r", flush=True)
        print(f"  [{kind}] {got}/{need} 완료")

    if plan.get("A"):
        emit("A", plan["A"], regs, lambda p: (
            fill(PROMPTS["A"], name=os.path.basename(p), body=read_text(p, 10000)),
            {"source_file": os.path.basename(p), "source_kind": "규정본문"},
        ))
    if plan.get("B"):
        emit("B", plan["B"], apps, lambda p: (
            fill(PROMPTS["B"], name=os.path.basename(p),
                 body=strip_md_header(read_text(p, 10000))),
            {"source_file": os.path.basename(p), "source_kind": "별표"},
        ))
    if plan.get("C"):
        def build_c(pair):
            rp, ap = pair
            rtxt = read_text(rp, 20000)
            atxt = strip_md_header(read_text(ap, 8000))
            if len(atxt) < 200 or not is_usable_appendix(ap, atxt):
                return None
            app_only = appendix_only_values(atxt, rtxt)
            if len(app_only) < 3:          # 별표 고유값이 없으면 연계 자체가 불가능
                return None
            return (
                fill(PROMPTS["C"],
                     reg_name=os.path.basename(rp),
                     reg_body=find_ref_article(rtxt, os.path.basename(ap)),
                     app_name=os.path.basename(ap), app_body=atxt),
                {"source_file": os.path.basename(rp),
                 "linked_file": os.path.basename(ap), "source_kind": "본문+별표"},
                lambda qa: c_type_is_linked(qa, app_only),
            )
        emit("C", plan["C"], pairs, build_c)
    if plan.get("D"):
        emit("D", plan["D"], nots, lambda p: (
            fill(PROMPTS["D"], name=os.path.basename(p), body=read_text(p, 8000)),
            {"source_file": os.path.basename(p), "source_kind": "공지"},
        ))

    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    global MODEL
    if not MODEL:
        try:
            with urllib.request.urlopen(VLLM.replace("/chat/completions", "/models"), timeout=10) as r:
                MODEL = json.loads(r.read().decode())["data"][0]["id"]
        except Exception as e:
            print("모델 조회 실패:", e); sys.exit(1)
    print("모델:", MODEL)

    plan = PLAN_PILOT if a.pilot else PLAN_FULL
    out = a.out or (os.path.join(ROOT, "rag_test_set_v2_pilot.json") if a.pilot
                    else os.path.join(ROOT, "rag_test_set_v2.json"))

    rows, stats = generate(plan)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\n생성 {len(rows)}건 -> {out}")
    print("\n[필터 통계]")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
