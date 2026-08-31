"""
title: Filesystem RAG (Agentic, No-Vector)
author: 연구기관 experiment
description: 벡터 DB/임베딩 없이, LLM이 연구기관 규정·별표·공지 폴더를 직접 탐색해서 답하는 filesystem-based agentic RAG 실험용 도구. list → grep → read 의 3단 흐름.
"""

import os
import re
from urllib.parse import quote


# === 연구기관 경로 시스템 (환경변수 CORPUS_ROOT로 통일) ===
CORPUS_ROOT = os.environ.get("CORPUS_ROOT", r"/data/regulations")

CATEGORY_DIRS = {
    "규정": os.path.join(CORPUS_ROOT, "regulation_txt_merged"),
    "별표": os.path.join(CORPUS_ROOT, "regulation_md_merged"),
    "공지": os.path.join(CORPUS_ROOT, "crawled_data", "analyzed"),
}

# PDF 서빙 경로(nginx /org-pdfs/...). 공지(공지사항)는 PDF가 없으므로 제외.
PDF_BASE = "/org-pdfs"
PDF_AVAILABLE_CATEGORIES = ("규정", "별표")

MAX_FILE_BYTES = 200_000  # 단일 파일 응답 상한 (LLM 컨텍스트 보호)


class Tools:
    def __init__(self):
        pass

    # ---------- 내부 유틸 ----------
    def _resolve_dirs(self, category: str):
        if category == "전체":
            return [(k, v) for k, v in CATEGORY_DIRS.items() if os.path.isdir(v)]
        if category in CATEGORY_DIRS and os.path.isdir(CATEGORY_DIRS[category]):
            return [(category, CATEGORY_DIRS[category])]
        return []

    def _strip_ext(self, name: str) -> str:
        return re.sub(r"\.(txt|md)$", "", name, flags=re.IGNORECASE)

    def _pdf_url(self, fname: str, cat: str) -> str:
        """원본 파일명 → PDF 서빙 URL. 공지는 PDF 없음."""
        if cat not in PDF_AVAILABLE_CATEGORIES:
            return ""
        name = self._strip_ext(fname)
        # 별표/별지/별첨은 원래 이름 그대로 + .pdf
        if any(kw in name for kw in ("별표", "별지", "별첨")):
            pdf_filename = name + ".pdf"
        else:
            # 규정 본문은 "_제N조..." 같은 조항 표시 제거 후 .pdf
            pdf_filename = re.sub(r"_제\s*\d+[조항].*$", "", name) + ".pdf"
        return f"{PDF_BASE}/{quote(pdf_filename, safe='')}"

    def _pdf_marker(self, fname: str, cat: str) -> str:
        """프론트가 PDF 아이콘으로 렌더링하는 {{PDF:url}} 마커."""
        url = self._pdf_url(fname, cat)
        return f" {{{{PDF:{url}}}}}" if url else ""

    def _read(self, fname: str, fpath: str, cat: str) -> str:
        try:
            size = os.path.getsize(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read(MAX_FILE_BYTES)
            note = ""
            if size > MAX_FILE_BYTES:
                note = f"\n[잘림: 전체 {size}바이트 중 앞 {MAX_FILE_BYTES}바이트만 표시]"
        except Exception as e:
            return f"파일 읽기 실패: {e}"
        header = f"📄 [{cat}] {fname}{self._pdf_marker(fname, cat)}"
        return header + "\n" + "=" * 50 + f"\n{content}\n" + "=" * 50 + note

    # ---------- 도구 1: 파일명 탐색 ----------
    def list_files(self, category: str = "전체", keyword: str = "", limit: int = 50) -> str:
        """
        [1단계 — 파일명 탐색]
        지정 카테고리 폴더의 파일명을 반환합니다. keyword가 주어지면 파일명에 해당 키워드가 포함된 것만 필터링됩니다.
        파일 수가 많으므로 keyword 없이 호출하지 마세요. 가능한 한 2~3글자 핵심어를 함께 넘기세요.

        Args:
            category: "규정" | "별표" | "공지" | "전체"
            keyword: 파일명 필터 키워드 (예: "출장", "여비", "복지")
            limit: 최대 반환 개수 (기본 50)

        Returns:
            "[카테고리] 파일명" 라인의 묶음. 매칭이 limit을 초과하면 잘립니다.
        """
        dirs = self._resolve_dirs(category)
        if not dirs:
            return f"카테고리 '{category}'를 찾을 수 없습니다."

        kw = keyword.strip()
        out, total = [], 0
        for cat, d in dirs:
            try:
                files = sorted(os.listdir(d))
            except Exception as e:
                out.append(f"[{cat}] 폴더 접근 실패: {e}")
                continue
            matched = [f for f in files if (not kw) or (kw in f)]
            total += len(matched)
            for f in matched[:limit]:
                out.append(f"[{cat}] {f}")
            if len(matched) > limit:
                out.append(f"... [{cat}] +{len(matched) - limit} more")

        if not out:
            return f"keyword '{keyword}'로 매칭된 파일이 없습니다. (category={category})"
        return "\n".join(out) + f"\n\n총 매칭: {total}건"

    # ---------- 도구 2: 본문 패턴 검색 ----------
    def grep_content(
        self,
        pattern: str,
        category: str = "전체",
        max_hits: int = 20,
        context_lines: int = 1,
    ) -> str:
        """
        [2단계 — 본문 패턴 검색]
        파일 내용에서 정규식/문자열 패턴을 찾아 매칭된 파일·라인·주변 컨텍스트를 반환합니다.
        파일명만으로 후보가 안 좁혀질 때 사용하세요.

        Args:
            pattern: 정규식 또는 단순 문자열 (예: "출장비.*지급", "복지포인트")
            category: "규정" | "별표" | "공지" | "전체"
            max_hits: 반환할 최대 hit 수 (기본 20)
            context_lines: 매칭 라인 위/아래 컨텍스트 라인 수 (기본 1)

        Returns:
            "[카테고리] 파일명 :라인번호: 스니펫" 형식.
        """
        dirs = self._resolve_dirs(category)
        if not dirs:
            return f"카테고리 '{category}'를 찾을 수 없습니다."

        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            return f"잘못된 정규식: {e}"

        hits = []
        for cat, d in dirs:
            try:
                fnames = sorted(os.listdir(d))
            except Exception:
                continue
            for fname in fnames:
                fpath = os.path.join(d, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except Exception:
                    continue
                for i, line in enumerate(lines):
                    if regex.search(line):
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        snippet = "".join(lines[start:end]).rstrip()
                        hits.append(f"[{cat}] {fname} :{i+1}:\n{snippet}\n---")
                        if len(hits) >= max_hits:
                            break
                if len(hits) >= max_hits:
                    break
            if len(hits) >= max_hits:
                break

        if not hits:
            return f"패턴 '{pattern}' 매칭 없음. (category={category})"
        tail = f"\n\n[최대 {max_hits}건까지 표시. 결과가 잘렸을 수 있음]" if len(hits) >= max_hits else ""
        return "\n".join(hits) + tail

    # ---------- 도구 3: 파일 정독 ----------
    def read_file(self, filename: str, category: str = "전체") -> str:
        """
        [3단계 — 파일 정독]
        후보 파일명을 받아 해당 파일 전체 내용을 반환합니다.
        list_files 또는 grep_content로 후보를 좁힌 다음 호출하세요.

        Args:
            filename: 확장자 포함/미포함, 전각 언더스코어 모두 지원
            category: "규정" | "별표" | "공지" | "전체"

        Returns:
            "📄 [카테고리] 파일명 === 본문 ===" 포맷.
        """
        dirs = self._resolve_dirs(category)
        if not dirs:
            return f"카테고리 '{category}'를 찾을 수 없습니다."

        target = filename.strip().replace("＿", "_")
        target_norm = re.sub(r"\s+", " ", target)
        target_base = self._strip_ext(target_norm)

        # 1차: 정확 매칭
        for cat, d in dirs:
            try:
                fnames = os.listdir(d)
            except Exception:
                continue
            for fname in fnames:
                fbase = self._strip_ext(fname)
                if fname == target or fbase == target_base or fname == target_base:
                    return self._read(fname, os.path.join(d, fname), cat)

        # 2차: 부분 매칭
        best, best_score = None, 0
        for cat, d in dirs:
            try:
                fnames = os.listdir(d)
            except Exception:
                continue
            for fname in fnames:
                fbase = self._strip_ext(fname)
                if target_base in fbase or fbase in target_base:
                    score = len(target_base)
                    if score > best_score:
                        best_score = score
                        best = (fname, os.path.join(d, fname), cat)
        if best:
            return self._read(*best)

        return f"'{filename}' 파일을 찾을 수 없습니다. list_files로 후보 다시 확인 권장."
