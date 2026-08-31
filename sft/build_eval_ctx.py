# -*- coding: utf-8 -*-
"""평가셋 394건에 gold context(근거 문서 원문)를 붙인 자급자족 평가 파일 생성"""
import os, re, json, glob

ROOT = "/data/regulations"
DIRS = ["regulation_txt_merged", "regulation_md_merged", "crawled_data/analyzed",
        "flattened_regulations"]
idx = {}
for d in DIRS:
    for p in glob.glob(os.path.join(ROOT, d, "*")):
        if os.path.isfile(p):
            idx.setdefault(os.path.basename(p), p)

def read(p, limit=9000):
    for enc in ("utf-8", "cp949"):
        try:
            return open(p, encoding=enc).read()[:limit]
        except UnicodeDecodeError:
            continue
        except Exception:
            return ""
    return ""

def find(name):
    name = os.path.basename(str(name or "")).strip()
    if name in idx:
        return idx[name]
    for ext in (".txt", ".md"):
        if name + ext in idx:
            return idx[name + ext]
    base = re.sub(r"\.(txt|md|pdf)$", "", name)
    for k in idx:
        if re.sub(r"\.(txt|md)$", "", k) == base:
            return idx[k]
    return None

rows = json.load(open(ROOT + "/scripts/rag_test_set_v2.json", encoding="utf-8"))
out, miss = [], 0
for r in rows:
    parts = []
    p = find(r.get("source_file"))
    if p:
        parts.append(read(p, 8000))
    lf = r.get("linked_file")
    if lf:
        p2 = find(lf)
        if p2:
            parts.append(read(p2, 6000))
    if not parts:
        miss += 1
        continue
    out.append({"type": r["type"], "question": r["question"], "gold": r["answer"],
                "source_file": r.get("source_file", ""), "context": "\n\n".join(parts)})

json.dump(out, open("/tmp/eval_ctx.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"생성 {len(out)}건 / 컨텍스트 미발견 {miss}건")
