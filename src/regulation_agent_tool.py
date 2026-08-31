"""
title: Regulation Agent Tool (연구기관 Expert)
author: antigravity
description: 연구기관(연구기관)의 규정, 지침, 공지사항을 검색하고 분석하는 '전용 전문가' 도구입니다. 모든 업무 관련 질문(인사, 복지, 연구비, 계약 등)에 대해 반드시 이 도구를 사용하여 최신 정보를 확인해야 합니다.
"""

import os
import re
import logging
import hashlib
import requests
import chromadb
import traceback
import asyncio
import aiohttp
from urllib.parse import quote
from chromadb.config import Settings


# === 문서 ID 체계 ===
#
# LLM 이 read_regulation_file 에 파일명을 문자열로 옮겨 적는 과정에서
# 전각 '＿', '.pdf' 확장자, 임의 번호 삽입, 공백 개수 차이 등 변형이 발생해
# 원문을 찾지 못하고 같은 호출을 반복하다 턴을 소진하는 문제가 있었다.
# (E2E 측정: 원문 직독 정확도 7.4% -> 경로 수정 후 37.9%)
#
# 파일명 대신 7자 토큰(D + md5 앞 6자)을 주고받게 해 문자열 매칭을 제거한다.
# ID 는 파일명에서 계산되는 결정적 값이므로 별도 저장이 필요 없고,
# 멀티 워커(UVICORN_WORKERS=8) 환경에서도 워커마다 동일한 값이 나온다.
_DOC_INDEX = None          # {doc_id: 절대경로} — 프로세스별 지연 생성 캐시
DOC_ID_RE = re.compile(r"^D[0-9a-f]{6}$")


def is_appendix_name(name: str) -> bool:
    return any(kw in str(name or "") for kw in ("별표", "별지", "별첨"))


def make_doc_id(name: str) -> str:
    """파일명 -> 문서 ID (확장자·전각·공백 차이를 흡수해 정규화 후 해시)"""
    base = os.path.basename(str(name or "")).strip()
    base = re.sub(r"\.(txt|md|pdf)$", "", base, flags=re.I)
    base = base.replace("＿", "_")
    base = re.sub(r"\s+", " ", base).strip().lower()
    return "D" + hashlib.md5(base.encode("utf-8")).hexdigest()[:6]

# 전역 클라이언트 캐싱 (ChromaDB 인스턴스 충돌 방지)
_CHROMA_CLIENT = None


# === 연구기관 경로 시스템 (환경변수 CORPUS_ROOT로 통일) ===
CORPUS_ROOT = os.environ.get("CORPUS_ROOT", r"/data/regulations")


class Tools:
    def __init__(self):
        global _CHROMA_CLIENT
        # 1. DB 경로 설정 (v2: 규정 + 별표 분리 DB / bulletin: 공지사항 DB)
        self.db_path = os.path.join(CORPUS_ROOT, "backend", "data", "vector_db2")
        self.notice_db_path = os.path.join(CORPUS_ROOT, "backend", "data", "bulletin")
        self.parent_org_db_path = os.path.join(CORPUS_ROOT, "backend", "data", "vector_parent_org")

        # 2. ChromaDB 클라이언트 초기화
        if _CHROMA_CLIENT is None:
            try:
                _CHROMA_CLIENT = chromadb.PersistentClient(
                    path=self.db_path,
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                print(f"[DEBUG] 규정 DB 연결 생성: {self.db_path}")
            except Exception as e:
                print(f"[DEBUG] 규정 DB 연결 실패: {e}")
                _CHROMA_CLIENT = chromadb.PersistentClient(path=self.db_path)

        self.client = _CHROMA_CLIENT

        # 2-1. 공지사항 전용 클라이언트 (경로가 다르므로 별도 인스턴스)
        try:
            self.notice_client = chromadb.PersistentClient(
                path=self.notice_db_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            print(f"[DEBUG] 공지사항 DB 연결 생성: {self.notice_db_path}")
        except Exception as e:
            print(f"[DEBUG] 공지사항 DB 연결 실패: {e}")
            self.notice_client = None

        # 2-2. 상위기관 전용 클라이언트
        try:
            self.parent_org_client = chromadb.PersistentClient(
                path=self.parent_org_db_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            print(f"[DEBUG] 상위기관 DB 연결 생성: {self.parent_org_db_path}")
        except Exception as e:
            print(f"[DEBUG] 상위기관 DB 연결 실패: {e}")
            self.parent_org_client = None

        # 3. 규정집(TXT) 디렉토리 설정 - 절대 경로 우선
        self.base_dirs = []
        # [2026-08 수정] 통합 원문(regulation_txt_merged)을 최우선으로 둔다.
        #   read_regulation_file 은 '규정 전체를 정독'하는 도구인데, 기존 목록에는
        #   조문 단위로 쪼갠 flattened_regulations 만 있어 전체 원문을 읽을 수 없었다.
        #   그 결과 "인사고과 시행기준"을 요청해도 부분 매칭으로 "..._부칙"(164자) 같은
        #   조각이 반환되고, LLM이 같은 호출을 반복하다 턴을 소진했다.
        #   (E2E 측정: 원문 직독 호출 27건 중 정답 2건)
        absolute_base_dirs = [
            os.path.join(CORPUS_ROOT, "regulation_txt_merged"),   # 규정 통합 원문 (최우선)
            os.path.join(CORPUS_ROOT, "regulation_md_merged"),    # 별표 통합 원문
            os.path.join(CORPUS_ROOT, "flattened_regulations"),   # 조문 단위 분할본
            os.path.join(CORPUS_ROOT, "marker_output_별표"),
            os.path.join(CORPUS_ROOT, "crawled_data", "analyzed"),
            os.path.join(CORPUS_ROOT, "parsed_regulations_parent_org"),
        ]
        for path in absolute_base_dirs:
            if os.path.exists(path) and os.path.isdir(path):
                self.base_dirs.append(path)
        print(f"[DEBUG] __init__ base_dirs: {self.base_dirs} (cwd: {os.getcwd()})")

        # 절대 경로로 못 찾은 경우 상대 경로 fallback
        if not self.base_dirs:
            folder_names = ["flattened_regulations", "marker_output_별표", "crawled_data/analyzed", "parsed_regulations_parent_org"]
            for folder_name in folder_names:
                candidates = [
                    os.path.join(os.getcwd(), "..", folder_name),
                    os.path.join(os.getcwd(), folder_name),
                    f"./data/vector_db/{folder_name}",
                ]
                try:
                    tool_dir = os.path.dirname(os.path.abspath(__file__))
                    backend_dir = os.path.dirname(tool_dir)
                    project_root = os.path.dirname(backend_dir)
                    candidates.insert(0, os.path.join(project_root, folder_name))
                except NameError:
                    pass
                for path in candidates:
                    try:
                        if os.path.exists(path) and os.path.isdir(path):
                            self.base_dirs.append(os.path.abspath(path))
                            break
                    except Exception:
                        continue

    def _doc_index(self, rebuild: bool = False) -> dict:
        """{문서ID: 절대경로} 역인덱스. 파일 목록만 훑으므로 가볍고, 디스크에 남기지 않는다."""
        global _DOC_INDEX
        if _DOC_INDEX is not None and not rebuild:
            return _DOC_INDEX
        idx = {}
        for base_dir in self.base_dirs:
            try:
                for root, _dirs, files in os.walk(base_dir):
                    for f in files:
                        # 먼저 등록된 디렉토리(통합 원문)가 우선
                        idx.setdefault(make_doc_id(f), os.path.join(root, f))
            except Exception:
                continue
        _DOC_INDEX = idx
        print(f"[DEBUG] 문서 ID 인덱스 생성: {len(idx)}건")
        return idx

    def read_regulation_file(self, filename: str) -> str:
        """
        [심화 검색] 특정 규정 파일의 '전체 내용'을 정독합니다.

        사용 지침:
        1. 먼저 `semantic_search`를 통해 관련 조항을 찾으세요.
        2. 검색된 조항만으로 답변이 부족하거나, 해당 규정 전체의 맥락 파악이 꼭 필요할 때만 이 도구를 사용하세요.
        3. **`semantic_search` 결과에 대괄호로 표시된 문서 ID(예: `[D3f2a1b]`)를 그대로 입력하세요.**
           파일명을 직접 타이핑하지 마십시오. 긴 한글 파일명을 옮겨 적다 오타가 나면 문서를 찾지 못합니다.
           (문서 ID를 모르는 경우에 한해 파일명을 넣어도 동작합니다.)
        """
        try:
            # --- 문서 ID 경로: 대괄호/공백을 제거하고 D+6자리 패턴이면 인덱스에서 직접 조회 ---
            raw = str(filename or "").strip().strip("[]").strip()
            if DOC_ID_RE.match(raw):
                idx = self._doc_index()
                path = idx.get(raw) or self._doc_index(rebuild=True).get(raw)
                if path and os.path.exists(path):
                    fname = os.path.basename(path)
                    base_dir = next((d for d in self.base_dirs
                                     if os.path.normpath(path).startswith(os.path.normpath(d))),
                                    os.path.dirname(path))
                    print(f"[DEBUG] 문서 ID '{raw}' -> {fname}")
                    for enc in ("utf-8", "cp949"):
                        try:
                            with open(path, "r", encoding=enc) as fp:
                                content = fp.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        return f"문서 ID '{raw}' 파일을 읽을 수 없습니다."
                    folder_type = "본문"
                    if "marker_output" in base_dir or "md_merged" in base_dir:
                        folder_type = "별표/서식"
                    if "crawled_data" in base_dir:
                        folder_type = "공지사항"
                    if "parent_org" in base_dir:
                        folder_type = "상위기관"
                    return (f"📄 문서: {fname} ({folder_type})\n" + "=" * 50
                            + "\n" + content + "\n" + "=" * 50)
                print(f"[DEBUG] 문서 ID '{raw}' 미발견 — 파일명 매칭으로 폴백")
            filename = filename.strip()
            # 전각 언더스코어(＿)를 반각(_)으로 변환 (LLM이 display_filename을 넘기는 경우 대응)
            filename = filename.replace("\uff3f", "_")
            print(
                f"[DEBUG] read_regulation_file 호출: '{filename}' (cwd: {os.getcwd()})"
            )

            # 항상 절대 경로를 보장 (exec() 환경에서 base_dirs가 비거나 cwd가 다를 수 있음)
            absolute_dirs = [
                os.path.join(CORPUS_ROOT, "regulation_txt_merged"),   # 통합 원문 (최우선)
                os.path.join(CORPUS_ROOT, "regulation_md_merged"),    # 별표 통합 원문
                os.path.join(CORPUS_ROOT, "flattened_regulations"),
                os.path.join(CORPUS_ROOT, "marker_output_별표"),
                os.path.join(CORPUS_ROOT, "crawled_data", "analyzed"),
                os.path.join(CORPUS_ROOT, "parsed_regulations_parent_org"),
                os.path.join(CORPUS_ROOT, "merged_regulations"),
            ]
            if not self.base_dirs:
                print("[DEBUG] base_dirs가 비어있어 절대 경로로 재설정")
                for path in absolute_dirs:
                    if os.path.exists(path):
                        self.base_dirs.append(path)
            else:
                # base_dirs가 있더라도 절대 경로가 누락되었으면 추가
                existing = set(os.path.normpath(d) for d in self.base_dirs)
                for path in absolute_dirs:
                    if os.path.normpath(path) not in existing and os.path.exists(path):
                        self.base_dirs.append(path)

            print(f"[DEBUG] 검색 대상 폴더 목록: {self.base_dirs}")

            def _read_and_return(file_path, file, base_dir):
                """파일 읽기 공통 함수 — 본문은 data-model-context div로 감싸 사용자에게는 숨기고 LLM에게만 전달"""
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                folder_type = "본문"
                if "marker_output" in base_dir:
                    folder_type = "별표/서식"
                if "crawled_data" in base_dir:
                    folder_type = "공지사항"
                if "parsed_regulations_parent_org" in base_dir:
                    folder_type = "상위기관"
                # 사용자에게는 간단한 한 줄만 보이고, LLM은 div 안의 전체 본문을 수신
                return (
                    f"📄 [{folder_type}] {file} 전체 본문을 불러왔습니다.\n"
                    f"<div data-model-context>\n"
                    f"[INTERNAL_DATA: {file} 전체 본문 ({folder_type})]\n"
                    f"{content}\n"
                    f"</div>"
                )

            clean_input = re.sub(r'\s+', ' ', filename.replace(".txt", "").replace(".md", "").replace(".pdf", "").strip())
            # 숫자+언더스코어 패턴 제거 (모델이 _07_ 같은 번호를 임의로 삽입하는 경우 대응)
            clean_input_normalized = re.sub(r'_\d+_', '_', clean_input)

            # 1차: 정확한 매칭
            for base_dir in self.base_dirs:
                for root, dirs, files in os.walk(base_dir):
                    for file in files:
                        clean_file = re.sub(r'\s+', ' ', file.replace(".txt", "").replace(".md", "").replace(".pdf", "").strip())
                        if clean_file == clean_input:
                            try:
                                return _read_and_return(os.path.join(root, file), file, base_dir)
                            except:
                                continue

            # 2차: 부분 매칭 (모델이 파일명을 부정확하게 생성한 경우 대응)
            #
            # [2026-08 수정] 기존에는 모든 후보의 점수가 len(clean_input) 으로 동일해
            # 먼저 발견된 파일이 그대로 채택됐다. 그 결과 "인사고과 시행기준"을 요청해도
            # 조각 파일("..._부칙", 164자)이 반환되어 LLM이 같은 호출을 반복했다.
            # 이제 (1) 조각이 아닌 통합 문서 (2) base_dirs 우선순위 (3) 길이 근접도
            # 순으로 점수를 매겨, 요청한 '규정 전체'가 선택되도록 한다.
            FRAGMENT = re.compile(r"_(제\s*\d+\s*조|부칙|서문|별표|별지)")

            best_match = None
            best_key = None
            for dir_rank, base_dir in enumerate(self.base_dirs):
                for root, dirs, files in os.walk(base_dir):
                    for file in files:
                        clean_file = re.sub(r'\s+', ' ', file.replace(".txt", "").replace(".md", "").replace(".pdf", "").strip())
                        clean_file_normalized = re.sub(r'_\d+_', '_', clean_file)
                        # 핵심 키워드가 포함되어 있는지 확인
                        if clean_input_normalized in clean_file_normalized or clean_file_normalized in clean_input_normalized:
                            input_is_fragment = bool(FRAGMENT.search(clean_input_normalized))
                            # 요청 자체가 조각(제N조/별표)이 아니라면 조각 파일은 뒤로 민다
                            is_fragment = bool(FRAGMENT.search(clean_file_normalized)) and not input_is_fragment
                            length_gap = abs(len(clean_file_normalized) - len(clean_input_normalized))
                            key = (is_fragment, dir_rank, length_gap)
                            if best_key is None or key < best_key:
                                best_key = key
                                best_match = (os.path.join(root, file), file, base_dir)

            if best_match:
                try:
                    print(f"[DEBUG] 부분 매칭으로 찾음: {best_match[1]}")
                    return _read_and_return(*best_match)
                except:
                    pass

            # 못 찾았을 때 디버깅 정보 반환
            return f"'{filename}' 파일을 찾을 수 없습니다.\n(검색 경로: {self.base_dirs}, cwd: {os.getcwd()})"
        except Exception as e:
            return f"오류 발생: {e}"

    async def semantic_search(
        self,
        query: str,
        collection: str = "전체",
        top_k: int = 3,
        threshold: float = 0.95, # 공지사항 검색을 위해 임계값 완화 (기본 0.85 -> 0.95)
    ) -> str:
        """
        [필수/최우선 사용] 본 도구는 연구기관(연구기관)의 규정과 관련된 질문에 대해 **가장 먼저** 호출되어야 하는 1차 검색 도구입니다.
        본인의 내부 지식(Internal Knowledge)은 절대 사용하지 마십시오. 규정은 수시로 개정되므로 반드시 이 도구의 검색 결과만을 기반으로 답변해야 합니다.
        
        [검색 연계 지침 - 필수 확인]
        1. 사용자의 모든 규정 관련 질문은 무조건 이 `semantic_search`를 가장 먼저 호출하여 연구기관 규정을 확인하십시오.
        2. 만약 이 도구로 검색을 했는데 **관련된 결과가 없거나 정보가 부족한 경우, 검색을 포기하지 말고 즉시 연이어 `parent_org_search` 도구를 호출**하여 상위기관 규정을 찾아보십시오.
        3. 두 도구를 모두 사용한 후에도 관련된 결과가 없을 때만 "검색 결과가 없습니다"라고 답변하십시오.
        4. [🚨절대 주의🚨] **이 도구(`semantic_search`)의 검색 결과로 답변을 작성할 때는 "상위기관 규정에 따르면..." 이라는 표현을 절대로 사용하지 마십시오.** 이 도구는 오직 자체 원내 규정만 다룹니다. 상위기관 언급은 `parent_org_search`의 결과에만 붙여야 합니다.
        5. [⚠️청크 한계] 본 도구는 각 매칭 결과의 **검색된 청크 조각만** 반환합니다. 표·양식이 포함된 별표, 금액·기준이 적힌 공지, 조항이 길어 잘렸을 가능성이 있는 본문 등 **청크만으로 정확한 답을 보장하기 어려운 경우 반드시 `read_regulation_file` 도구를 추가 호출하여 해당 파일의 전체 본문을 확인한 뒤 답변하십시오.**

        [호출 대상 예시 - 아래 키워드 포함 시 반드시 호출]
        - 인사/복지: 연차, 휴가, 출장, 여비, 수당, 급여, 채용, 평가, 복지포인트 등
        - 업무/행정: 계약, 구매, 시스템 사용법(ERP 등), 시설 예약, 식단, 보안 등
        - 연구/사업: 연구비, 과제 관리, 지식재산권, 기술료, 논문 가점 등

        [응답 스타일 지침 - 반드시 준수]
        1. 시각 중심 구성: 제목과 주요 항목 앞에 적절한 이모지(🏆, ✅, 📌, 💡)를 사용하세요.
        2. 표(Table) 활용: 배점, 기준, 금액 등 수치가 포함된 데이터는 무조건 마크다운 표로 작성하세요.
        3. 구체적 예시: 산정 방식이 복잡할 경우 반드시 '산정 예시' 섹션을 추가하여 사용자 이해를 도우세요.
        4. 구조화: [개요] - [상세 내용(표 추천)] - [핵심 요약] - [출처] 순서로 답변을 구성하세요.
        5. 출처 명시: 답변 하단에 [관련 규정/지침 명칭 및 조항]을 명확히 기재하세요.
        """
        print(
            f"[DEBUG] semantic_search 호출됨: query='{query}', collection='{collection}'"
        )
        try:
            # 0. 쿼리 설정 (멀티쿼리 확장 비활성화 - 원문 쿼리만 사용)
            search_queries = [query]
            # [비활성화] 멀티쿼리 확장 (descriptive + HyDE)
            # 속도 향상 및 검색 정확도 개선을 위해 원문 쿼리만 사용
            # 재활성화 시 아래 주석을 해제하세요.
            # ---
            # async def expand_query_task(session, task_type):
            #     ... (생략)
            # async with aiohttp.ClientSession() as session:
            #     tasks = [expand_query_task(session, "descriptive"), expand_query_task(session, "hyde")]
            #     expanded_results = await asyncio.gather(*tasks)
            #     for res in expanded_results:
            #         if res and res not in search_queries:
            #             search_queries.append(res)
            # ---

            # 1. 임베딩 생성 (RAG 서버 11435 사용)
            all_embeddings = []
            async def get_embedding_task(session, q):
                try:
                    async with session.post(
                        "http://127.0.0.1:21435/api/embeddings",
                        json={"model": "qwen3-embedding:8b", "prompt": q},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("embedding")
                except Exception as e:
                    print(f"[DEBUG] 임베딩 생성 오류: {e}")
                return None

            async with aiohttp.ClientSession() as session:
                emb_tasks = [get_embedding_task(session, q) for q in search_queries]
                emb_results = await asyncio.gather(*emb_tasks)
                all_embeddings = [e for e in emb_results if e]

            if not all_embeddings:
                return "임베딩 생성에 실패했습니다."

            # 2. 검색 대상 컬렉션 결정 — 항상 전체 통합 검색
            #
            # [2026-08 변경] 키워드 기반 컬렉션 라우팅을 제거했다.
            #
            # 측정 근거 (테스트셋 394건, Recall@5):
            #     라우팅 ON  75.6%  ->  라우팅 OFF  85.5%   (+9.9%p)
            #     A 규정본문 80.0->90.7 / B 별표 80.7->91.4 / C 연계 60.8->71.6 / D 공지 70.0->72.5
            #     즉 공지 질의(D)를 포함한 '모든' 유형에서 전체 검색이 더 나았다.
            #
            # 실패 원인:
            #     "계약"(22건), "교육"(7건), "시스템"(5건), "공사"(4건) 처럼 규정에도 흔한 단어가
            #     공지 키워드에 포함되어 있었다. 그 결과 규정 질의 44건이 공지 컬렉션만 검색하고
            #     정답이 있는 규정/별표는 아예 조회하지 못하는 구조적 실패가 발생했다.
            #     (예: "계약직원 재임용 심사 기준" -> '계약' 매칭 -> 공지사항만 검색 -> 실패)
            #
            # 원래 우려했던 '공지사항이 규정 답변을 오염시키는 문제'는
            # 거리(distance) 기준 정렬과 threshold 필터로 충분히 걸러지는 것으로 확인되었다.
            # 검색 범위를 미리 배제해서 얻는 이득보다, 정답 컬렉션을 통째로 건너뛰는 손실이 컸다.
            collections_to_search = [
                ("regulations", "규정", self.client),
                ("appendices", "별표/서식", self.client),
                ("notices", "공지사항", self.notice_client),
            ]

            print(
                f"[DEBUG] 검색 대상 컬렉션 목록: {[c[0] for c in collections_to_search]}"
            )
            print(f"[DEBUG] 최종 쿼리 목록: {search_queries}")

            # 4. 모든 컬렉션에서 검색 수행 및 중복 제거
            results = {}
            for coll_name, coll_type, coll_client in collections_to_search:
                if coll_client is None: continue
                try:
                    coll = coll_client.get_collection(name=coll_name)
                    for emb in all_embeddings:
                        res = coll.query(
                            query_embeddings=[emb],
                            n_results=top_k * 2,
                            include=["documents", "distances", "metadatas"],
                        )
                        if res["documents"] and res["documents"][0]:
                            for i in range(len(res["documents"][0])):
                                content = res["documents"][0][i]
                                dist = res["distances"][0][i]
                                if dist <= threshold:
                                    if (
                                        content not in results
                                        or dist < results[content]["dist"]
                                    ):
                                        # [SAFE] metadata가 None이거나 원소가 None인 경우 빈 dict로
                                        meta_val = {}
                                        try:
                                            metas_list = res.get("metadatas") if isinstance(res, dict) else None
                                            if metas_list and metas_list[0] and i < len(metas_list[0]):
                                                meta_val = metas_list[0][i] or {}
                                                if not isinstance(meta_val, dict):
                                                    meta_val = {}
                                        except Exception:
                                            meta_val = {}
                                        results[content] = {
                                            "doc": content,
                                            "dist": dist,
                                            "meta": meta_val,
                                            "coll_type": coll_type,
                                        }
                except Exception as e:
                    print(f"[DEBUG] 컬렉션 '{coll_name}' 검색 오류: {e}")
                    continue

            sorted_results = sorted(results.values(), key=lambda x: x["dist"])[:top_k]
            print(f"[DEBUG] 검색 결과 건수: {len(results)}, 필터링 후(top_k): {len(sorted_results)}")
            if sorted_results:
                print(f"[DEBUG] 최상위 결과 거리: {sorted_results[0]['dist']:.4f}")

            # 4-1. GraphRAG 확장 — 검색 결과를 시드로 '인용 관계' 문서를 보강한다.
            #
            # 규정 본문과 별표는 서로를 명시적으로 인용하지만("...별표 제3호에 따라 지급한다"),
            # 벡터 검색은 두 문서를 낱개로만 보기 때문에 한쪽만 걸리는 경우가 많다.
            # 유사도를 다시 계산하지 않고 문서에 이미 적힌 인용 관계를 따라가므로,
            # 질의와 어휘가 전혀 겹치지 않는 수치표도 확실히 끌어올 수 있다.
            #
            # [2026-08 측정] 연계형 질의 74건 기준 별표 78.4% vs 본문 64.9%.
            #   병목이 '본문 조문 누락'이므로 별표->본문 역방향 확장이 특히 중요하다.
            #   (CitationGraph 내부에서 역방향에 최우선 순위를 부여)
            #
            # 확장 결과는 벡터 거리값이 없어 유사도 정렬에 섞을 수 없다.
            # 따라서 sorted_results 와 합치지 않고 별도 섹션으로 출력한다.
            graph_results = []
            try:
                from open_webui.utils.citation_graph import CitationGraph

                seed_metas = [item.get("meta") or {} for item in sorted_results]
                graph_results = CitationGraph().get_related_documents(seed_metas, limit=2)
                if graph_results:
                    print(f"[DEBUG] GraphRAG 확장: {len(graph_results)}건 "
                          f"({[r['metadata']['name'][:30] for r in graph_results]})")
            except Exception as e:
                # 확장 실패는 검색 자체를 막지 않는다 (보조 기능)
                print(f"[DEBUG] GraphRAG 확장 건너뜀: {type(e).__name__}: {e}")
                graph_results = []

            if not sorted_results:
                return f"'{collection}' 검색 결과 '{query}'와 관련된 내용을 찾을 수 없습니다."

            # 5. 출력 포맷팅 (섹션별 분리 및 공지사항 PDF 링크 생략)
            query_list_str = ", ".join([f"'{q}'" for q in search_queries])
            
            # 결과 분류
            reg_results = [r for r in sorted_results if r.get("coll_type") in ["규정", "별표/서식", "상위기관"]]
            notice_results = [r for r in sorted_results if r.get("coll_type") == "공지사항"]

            output = f"🔍 통합 검색 결과\n"
            output += f"**[검색어 목록]**: {query_list_str}\n\n"

            # 규정 및 지침 섹션
            if reg_results:
                output += "#### 📜 규정 및 지침\n"
                for i, item in enumerate(reg_results, 1):
                    doc = item.get("doc") or ""
                    score = (1 - item.get("dist", 1.0)) * 100
                    meta = item.get("meta") or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    source = meta.get("source", "Unknown")
                    filename = os.path.basename(source)
                    
                    # 변수 정의
                    name_without_ext = filename.replace(".txt", "").replace(".md", "")
                    display_filename = filename.replace("_", "＿")
                    coll_type = item.get("coll_type", "규정")

                    # 원문 전체가 필요할 때 LLM이 read_regulation_file 에 넘길 식별자.
                    # 조문 조각(_제N조)이 아니라 '규정 전체'를 가리키도록 접미어를 떼고 계산한다.
                    doc_root = re.sub(r"_(제\s*\d+\s*[조항].*|부칙.*|서문.*)$", "", name_without_ext)
                    doc_id = make_doc_id(doc_root if not is_appendix_name(name_without_ext)
                                         else name_without_ext)
                    
                    # [BUG FIX] 상위기관 문서인지 별표/서식인지 명확히 구분
                    is_parent_org = meta.get("is_parent_org", False)
                    if is_parent_org and coll_type != "상위기관":
                         coll_type = "상위기관"
                    
                    # PDF 파일명 생성 (조항 번호 제거)
                    is_appendix = any(kw in name_without_ext for kw in ["별표", "별지", "별첨"])
                    if is_appendix:
                        pdf_filename = name_without_ext + ".pdf"
                    else:
                        pdf_filename = re.sub(r"_제\s*\d+[조항].*$", "", name_without_ext) + ".pdf"
                    
                    pdf_url = f"/org-pdfs/{quote(pdf_filename, safe='')}"
                    
                    if is_appendix:
                        # 별표/서식: PDF 아이콘 + 청크 조각만 표시 (전체 본문 주입 제거 — 컨텍스트 절약)
                        # 표/양식 전체가 필요하면 LLM이 read_regulation_file을 별도 호출하도록 유도.
                        output += f"**{i}. [{coll_type}] {display_filename} ({score:.1f}%)** `[{doc_id}]` {{{{PDF:{pdf_url}}}}}\n"
                        output += f'\n<div data-model-context>\n[INTERNAL_DATA: {display_filename} 검색 청크 | 원문 전체는 read_regulation_file("{doc_id}") 로 조회]\n{doc.strip()}\n</div>\n\n'
                    else:
                        # 규정 본문: PDF 아이콘 + 청크 인용구
                        output += f"**{i}. [{coll_type}] {display_filename} ({score:.1f}%)** `[{doc_id}]` {{{{PDF:{pdf_url}}}}}\n"
                        output += f'\n> ' + doc.strip().replace("\n", "\n> ") + '\n\n'

            # 공지사항 섹션
            if notice_results:
                if reg_results:
                    output += "---\n"
                output += "#### 📢 사내 공지사항\n"
                for i, item in enumerate(notice_results, 1):
                    doc = item.get("doc") or ""
                    score = (1 - item.get("dist", 1.0)) * 100
                    meta = item.get("meta") or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    source = meta.get("source", "Unknown")
                    filename = os.path.basename(source)
                    # [BUG FIX] display_filename(전각) 대신 name_without_ext(원본) 사용
                    name_without_ext_notice = filename.replace(".txt", "").replace(".md", "")
                    display_filename_notice = filename.replace("_", "＿") # 공지사항 제목도 전각으로 표시
                    
                    # 공지사항은 PDF 링크 제외 — 청크 조각만 모델에 전달 (전체 본문 주입 제거)
                    # 공지 전체가 필요하면 LLM이 read_regulation_file을 별도 호출하도록 유도.
                    output += f"**{i}. {display_filename_notice} ({score:.1f}%)**\n"
                    output += f'\n<div data-model-context>\n[INTERNAL_DATA: {display_filename_notice} 검색 청크]\n{doc.strip()}\n</div>\n\n'

            # GraphRAG 확장 섹션 — 인용 관계로 따라온 보조 근거
            # 유사도 점수가 없으므로 위 검색 결과와 섞지 않고 별도로 표시한다.
            if graph_results:
                output += "---\n"
                output += "#### 🔗 관련 조문·별표 (인용 관계 기반)\n"
                output += "> 위 검색 결과가 인용하고 있거나, 위 결과를 인용하는 문서입니다. "
                output += "조문의 적용 대상·조건과 별표의 구체적 수치를 함께 확인하세요.\n\n"
                for i, item in enumerate(graph_results, 1):
                    gname = item["metadata"]["name"]
                    gdisplay = gname.replace("_", "＿")
                    gbody = (item["page_content"] or "").strip()
                    if len(gbody) > 2000:
                        gbody = gbody[:2000] + "\n...(이하 생략)"
                    output += f"**{i}. {gdisplay}**\n"
                    output += (f'\n<div data-model-context>\n'
                               f'[INTERNAL_DATA: {gdisplay} 인용관계 확장]\n{gbody}\n</div>\n\n')

            return output

        except Exception as e:
            error_msg = f"벡터 검색 시스템 오류: {e}\n\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            return error_msg

    async def parent_org_search(
        self,
        query: str,
        collection: str = "전체",
        top_k: int = 3,
        threshold: float = 0.95, 
    ) -> str:
        """
        [상위기관 규정 전용 검색] 상위기관인 상위기관(상위기관)의 규정, 지침, 요령 등을 검색할 때 사용하는 **2차 확장 검색 도구**입니다.
        
        [검색 연계 지침 - 필수 확인] 
        1. 이 도구는 단독으로 가장 먼저 실행되지 않아야 합니다. 반드시 `semantic_search`를 먼저 실행하십시오.
        2. `semantic_search`의 검색 결과에서 사용자의 질문에 대한 충분한 답변을 찾지 못했다면, 검색 완료된 것으로 간주하지 말고 **반드시 이 `parent_org_search` 도구를 추가로 호출하여 답변을 시도**하십시오.
        3. 사용자가 "상위기관", "상위기관" 등을 직접 언급한 경우에도 1차 `semantic_search` 이후 2차로 이 도구를 호출하여 종합적으로 판단하십시오.

        [응답 스타일 지침 - 반드시 준수]
        1. 시각 중심 구성: 제목과 주요 항목 앞에 적절한 이모지(🏆, ✅, 📌, 💡)를 사용하세요.
        2. 상위기관 명시: 반드시 답변 서두나 출처에 "상위기관 규정에 따르면~" 이라고 명확히 안내하세요.
        3. 표(Table) 활용: 수치가 포함된 데이터는 무조건 마크다운 표로 작성하세요.
        4. 출처 명시: 답변 하단에 [관련 규정 명칭 및 조항]을 기재하세요.
        """
        print(f"[DEBUG] parent_org_search 호출됨: query='{query}', collection='{collection}'")
        try:
            if not self.parent_org_client:
                return "상위기관 검색 시스템이 아직 연결되지 않았습니다."

            search_queries = [query]
            all_embeddings = []
            
            async def get_embedding_task(session, q):
                try:
                    async with session.post(
                        "http://127.0.0.1:21435/api/embeddings",
                        json={"model": "qwen3-embedding:8b", "prompt": q},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("embedding")
                except Exception as e:
                    print(f"[DEBUG] 임베딩 생성 오류 (상위기관): {e}")
                return None

            async with aiohttp.ClientSession() as session:
                emb_tasks = [get_embedding_task(session, q) for q in search_queries]
                emb_results = await asyncio.gather(*emb_tasks)
                all_embeddings = [e for e in emb_results if e]

            if not all_embeddings:
                return "임베딩 생성에 실패했습니다."

            collections_to_search = [("parent_org_regulations", "상위기관", self.parent_org_client)]
            results = {}
            for coll_name, coll_type, coll_client in collections_to_search:
                try:
                    coll = coll_client.get_collection(name=coll_name)
                    for emb in all_embeddings:
                        res = coll.query(
                            query_embeddings=[emb],
                            n_results=top_k * 2,
                            include=["documents", "distances", "metadatas"],
                        )
                        if res["documents"] and res["documents"][0]:
                            for i in range(len(res["documents"][0])):
                                content = res["documents"][0][i]
                                dist = res["distances"][0][i]
                                if dist <= threshold:
                                    if content not in results or dist < results[content]["dist"]:
                                        results[content] = {
                                            "doc": content,
                                            "dist": dist,
                                            "meta": res["metadatas"][0][i] if res["metadatas"] else {},
                                            "coll_type": coll_type,
                                        }
                except Exception as e:
                    print(f"[DEBUG] 컬렉션 '{coll_name}' 검색 오류: {e}")

            sorted_results = sorted(results.values(), key=lambda x: x["dist"])[:top_k]
            
            if not sorted_results:
                return f"상위기관 데이터베이스에서 '{query}'와 관련된 내용을 찾을 수 없습니다."

            query_list_str = ", ".join([f"'{q}'" for q in search_queries])
            output = f"🔍 상위기관 통합 검색 결과\n"
            output += f"**[검색어 목록]**: {query_list_str}\n\n"
            output += "#### 📜 상위기관 규정 및 지침\n"
            
            for i, item in enumerate(sorted_results, 1):
                doc = item["doc"]
                score = (1 - item["dist"]) * 100
                meta = item["meta"]
                source = meta.get("source", "Unknown")
                filename = os.path.basename(source)
                
                name_without_ext = filename.replace(".txt", "").replace(".md", "")
                display_filename = filename.replace("_", "＿")
                coll_type = "상위기관"
                
                is_appendix = any(kw in name_without_ext for kw in ["별표", "별지", "별첨"])
                if is_appendix:
                    pdf_filename = name_without_ext + ".pdf"
                else:
                    pdf_filename = re.sub(r"_제\s*\d+[조항].*$", "", name_without_ext) + ".pdf"
                
                pdf_url = f"/org-pdfs/{quote(pdf_filename, safe='')}"
                
                if is_appendix:
                    output += f"**{i}. [{coll_type}] {display_filename} ({score:.1f}%)** {{{{PDF:{pdf_url}}}}}\n"
                    full_appendix_content = self.read_regulation_file(name_without_ext)
                    if "파일을 찾을 수 없습니다" not in full_appendix_content:
                            content_to_show = full_appendix_content.split("=" * 50)[1].strip() if "=====" in full_appendix_content else full_appendix_content.strip()
                    else:
                            content_to_show = doc.strip()
                    output += f'\n<div data-model-context>\n[INTERNAL_DATA: {display_filename} 전체 내용]\n{content_to_show}\n</div>\n\n'
                else:
                    output += f"**{i}. [{coll_type}] {display_filename} ({score:.1f}%)** {{{{PDF:{pdf_url}}}}}\n"
                    output += f'\n> ' + doc.strip().replace("\n", "\n> ") + '\n\n'

            return output

        except Exception as e:
            error_msg = f"상위기관 벡터 검색 시스템 오류: {e}\n\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            return error_msg
