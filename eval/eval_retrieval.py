#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연구기관 RAG 검색 품질 평가 (Retrieval Evaluation)

테스트셋의 각 질문을 실제 검색 파이프라인에 통과시켜,
'정답 근거 문서'가 상위 k개 안에 들어오는지 측정한다.

측정 지표
  - Recall@k       : 정답 문서가 top-k 안에 있는 비율
  - C유형 분해      : 본문만 / 별표만 / 둘 다 맞춘 비율
                     -> "본문은 찾는데 별표를 못 찾는다"는 가설의 정량 검증
  - 라우팅 비교     : 키워드 라우팅 적용 vs 전체 컬렉션 검색

컨테이너 안에서 실행:
  docker exec org-openwebui python ./data/eval_retrieval.py \
      --testset ./data/rag_test_set_v2.json --topk 5
"""
import os, re, json, time, argparse, urllib.request, sys
from collections import defaultdict

EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:21435/api/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:8b")
DB_MAIN = os.environ.get("DB_MAIN", "./data/vector_db/vector_db2")
DB_NOTICE = os.environ.get("DB_NOTICE", "./data/vector_db/bulletin")

COL_REG = "regulations"
COL_APP = "appendices"
COL_NOT = "notices"

# regulation_agent_tool.py 의 라우팅 규칙과 동일하게 유지
NOTICE_KW = ["공지", "안내", "식단", "전진", "공사", "모집", "사례", "지적",
             "개선", "계약", "시스템", "변경", "설문", "교육", "간담회"]
REG_KW = ["규정", "지침", "요령", "기준", "조항", "제N조"]

PREFIX = re.compile(r"^(규정|지침|요령|기타|공지|별표)_")
CLASSNO = re.compile(r"^\d+\.\s*[^_]+_")          # "2. 인사_" 같은 분류 접두어
EXT = re.compile(r"\.(txt|md|pdf|json)$", re.I)


def norm_doc(name: str) -> str:
    """문서 식별자 정규화 - 접두어/분류/확장자/공백 차이를 흡수"""
    n = os.path.basename(str(name or "")).strip()
    n = EXT.sub("", n)
    n = re.sub(r"_analyzed$", "", n)
    prev = None
    while prev != n:                      # "지침_2. 인사_..." 처럼 중첩된 접두어 제거
        prev = n
        n = PREFIX.sub("", n)
        n = CLASSNO.sub("", n)
    n = re.sub(r"\s+", " ", n).strip().lower()
    return n


def base_reg_of_appendix(app_name: str) -> str:
    """'X_별표 제1호.md' -> 'X' (본문 문서명)"""
    n = EXT.sub("", os.path.basename(str(app_name or "")))
    return norm_doc(re.sub(r"_별표\s*제?[\d\-]+\s*호?$", "", n))


def embed(text, retries=3):
    body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(EMBED_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                j = json.loads(r.read().decode())
            e = j.get("embedding") or j.get("embeddings")
            if e:
                return e
        except Exception as ex:
            if i == retries - 1:
                print(f"  [임베딩 실패] {type(ex).__name__}", file=sys.stderr)
    return None


def route(question: str, force_all=False):
    """질의 성격에 따라 검색 대상 컬렉션 결정 (운영 코드와 동일 규칙)"""
    if force_all:
        return [COL_REG, COL_APP, COL_NOT]
    q = question.lower()
    is_notice = any(k in q for k in NOTICE_KW)
    is_reg = any(k in q for k in REG_KW)
    if is_notice and not is_reg:
        return [COL_NOT]
    if is_reg and not is_notice:
        return [COL_REG, COL_APP]
    return [COL_REG, COL_APP, COL_NOT]


ARTICLE_SUFFIX = re.compile(r"_(제\d+조|부칙|서문).*$")   # "_제5조(세부기준)" 처럼 뒤에 괄호가 붙는 형태까지


def strip_article(name: str) -> str:
    """조문 단위 파일명에서 문서명만 남긴다. (규정_X_제5조(...)  ->  X)"""
    return norm_doc(ARTICLE_SUFFIX.sub("", EXT.sub("", os.path.basename(str(name or "")))))


def hit_ids(metas):
    """검색 결과 메타데이터 -> 매칭용 식별자 집합"""
    docs, apps, nots = set(), set(), set()
    for m in metas:
        src = str(m.get("source", "") or "")
        dn = str(m.get("doc_name", "") or "")
        is_app = str(m.get("is_appendix", "")).lower() == "true" or "별표" in src
        if is_app:
            apps.add(norm_doc(src))
            if dn:
                apps.add(norm_doc(dn))
        elif m.get("title") or m.get("dept"):        # 공지 컬렉션 메타 특징
            nots.add(norm_doc(src))
        else:
            if dn:
                docs.add(norm_doc(dn))
            if src:
                docs.add(base_reg_of_appendix(src) if "별표" in src else strip_article(src))
    return docs, apps, nots


def judge(row, docs, apps, nots):
    """유형별 정답 판정. C는 본문/별표를 분리해서 본다."""
    t = row["type"]
    src = norm_doc(row["source_file"])
    if t == "A":
        return {"hit": src in docs}
    if t == "B":
        tgt = norm_doc(row["source_file"])
        return {"hit": tgt in apps}
    if t == "D":
        return {"hit": norm_doc(row["source_file"]) in nots or norm_doc(row["source_file"]) in docs}
    # C: 본문 + 별표 둘 다 필요
    a_ok = norm_doc(row.get("linked_file", "")) in apps
    r_ok = src in docs
    return {"hit": a_ok and r_ok, "reg_hit": r_ok, "app_hit": a_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="./data/rag_test_set_v2.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (스모크 테스트)")
    ap.add_argument("--no-routing", action="store_true", help="라우팅 없이 전체 검색")
    ap.add_argument("--graph", action="store_true",
                    help="GraphRAG 확장 포함 (인용 관계로 이웃 문서 보강)")
    ap.add_argument("--graph-limit", type=int, default=2, help="그래프 확장 문서 수")
    ap.add_argument("--out", default="./data/rag_eval_report.json")
    a = ap.parse_args()

    import chromadb
    from chromadb.config import Settings

    rows = json.load(open(a.testset, encoding="utf-8"))
    if a.limit:
        rows = rows[:a.limit]

    cli_main = chromadb.PersistentClient(path=DB_MAIN, settings=Settings(anonymized_telemetry=False))
    cli_not = chromadb.PersistentClient(path=DB_NOTICE, settings=Settings(anonymized_telemetry=False))
    cols = {
        COL_REG: cli_main.get_collection(COL_REG),
        COL_APP: cli_main.get_collection(COL_APP),
        COL_NOT: cli_not.get_collection(COL_NOT),
    }
    kgraph = None
    if a.graph:
        # open_webui 패키지는 backend/ 아래에 있으므로 import 경로를 보강한다
        for cand in (".", os.path.join(os.getcwd(), "backend"), os.getcwd()):
            if os.path.isdir(os.path.join(cand, "open_webui")) and cand not in sys.path:
                sys.path.insert(0, cand)
        from open_webui.utils.citation_graph import CitationGraph
        kgraph = CitationGraph()
        n_app = sum(1 for n in kgraph.graph.nodes if kgraph._is_appendix(n))
        print(f"GraphRAG: 노드 {kgraph.graph.number_of_nodes():,} "
              f"(별표 {n_app:,}) / 엣지 {kgraph.graph.number_of_edges():,}")

    print(f"평가 대상 {len(rows)}건 | top-k={a.topk} | "
          f"라우팅={'OFF' if a.no_routing else 'ON'} | "
          f"그래프={'ON' if a.graph else 'OFF'}")

    stat = defaultdict(int)
    cstat = defaultdict(int)
    details, t0 = [], time.time()

    for i, row in enumerate(rows, 1):
        q = row["question"]
        emb = embed(q)
        if emb is None:
            stat[f"{row['type']}:err"] += 1
            continue
        metas = []
        for cname in route(q, a.no_routing):
            try:
                res = cols[cname].query(query_embeddings=[emb], n_results=a.topk,
                                        include=["metadatas", "distances"])
                if res["metadatas"] and res["metadatas"][0]:
                    metas.extend(res["metadatas"][0])
            except Exception as e:
                print(f"  [{cname}] 검색 오류: {type(e).__name__}", file=sys.stderr)
        # GraphRAG 확장 — 벡터 검색 결과를 시드로 인용 관계 문서를 보강
        # (운영 코드 semantic_search 와 동일한 호출 방식)
        if kgraph is not None and metas:
            try:
                for gd in kgraph.get_related_documents(metas, limit=a.graph_limit):
                    metas.append(gd["metadata"])
            except Exception as e:
                print(f"  [graph] 확장 실패: {type(e).__name__}", file=sys.stderr)

        docs, apps, nots = hit_ids(metas)
        r = judge(row, docs, apps, nots)

        t = row["type"]
        stat[f"{t}:total"] += 1
        stat[f"{t}:hit"] += 1 if r["hit"] else 0
        if t == "C":
            cstat["total"] += 1
            cstat["reg"] += 1 if r.get("reg_hit") else 0
            cstat["app"] += 1 if r.get("app_hit") else 0
            cstat["both"] += 1 if r["hit"] else 0
            if r.get("reg_hit") and not r.get("app_hit"):
                cstat["reg_only"] += 1
        details.append({**{k: row[k] for k in ("type", "source_file", "question")},
                        "linked_file": row.get("linked_file", ""),
                        **r})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 리포트 ----------------
    print("\n" + "=" * 58)
    print(f"검색 품질 평가 결과  (top-k={a.topk}, 라우팅={'OFF' if a.no_routing else 'ON'})")
    print("=" * 58)
    labels = {"A": "규정본문", "B": "별표", "C": "본문+별표 연계", "D": "공지"}
    tot = hit = 0
    for t in ["A", "B", "C", "D"]:
        n, h = stat[f"{t}:total"], stat[f"{t}:hit"]
        if not n:
            continue
        tot += n; hit += h
        print(f"  [{t}] {labels[t]:14s} Recall@{a.topk} = {100*h/n:5.1f}%  ({h}/{n})")
    if tot:
        print(f"  {'전체':>19s} Recall@{a.topk} = {100*hit/tot:5.1f}%  ({hit}/{tot})")

    if cstat["total"]:
        c = cstat
        print("\n  [C유형 분해] — '별표를 못 찾는다' 가설 검증")
        print(f"    본문 검색 성공      : {100*c['reg']/c['total']:5.1f}%  ({c['reg']}/{c['total']})")
        print(f"    별표 검색 성공      : {100*c['app']/c['total']:5.1f}%  ({c['app']}/{c['total']})")
        print(f"    둘 다 성공          : {100*c['both']/c['total']:5.1f}%  ({c['both']}/{c['total']})")
        print(f"    본문만 (별표 누락)  : {100*c['reg_only']/c['total']:5.1f}%  ({c['reg_only']}/{c['total']})  <- GraphRAG 대상")

    errs = sum(v for k, v in stat.items() if k.endswith(":err"))
    if errs:
        print(f"\n  임베딩 실패: {errs}건")
    print(f"\n  소요 {time.time()-t0:.0f}초")

    json.dump({"topk": a.topk, "routing": not a.no_routing,
               "summary": dict(stat), "c_breakdown": dict(cstat),
               "details": details},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  상세 -> {a.out}")


if __name__ == "__main__":
    main()
