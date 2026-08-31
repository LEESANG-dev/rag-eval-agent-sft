import os
import re
import glob
import logging
import networkx as nx

log = logging.getLogger(__name__)

class CitationGraph:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CitationGraph, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return
        
        # Try to locate the data directories
        self.base_path = os.environ.get("CORPUS_ROOT") or os.getcwd()
        if os.path.basename(self.base_path) == 'backend':
            self.base_path = os.path.dirname(self.base_path)

        self.reg_dir = os.path.join(self.base_path, "flattened_regulations")

        # [2026-08 수정] 기존 경로 'flattened_appendices_marker'는 비어 있어
        # 별표 노드가 하나도 적재되지 않았고, 그 결과 조문<->별표 엣지가 생성되지 않았다.
        # 실제 별표 원문이 있는 디렉토리로 교체한다. (filesystem_rag_tool과 동일 경로)
        self.app_dir = os.path.join(self.base_path, "regulation_md_merged")
        if not os.path.isdir(self.app_dir):
            for cand in ("flattened_appendices_marker", "marker_output_별표"):
                p = os.path.join(self.base_path, cand)
                if os.path.isdir(p) and os.listdir(p):
                    self.app_dir = p
                    break
        
        self.graph = nx.DiGraph()
        self.documents = {} 
        self.doc_titles = {}
        self.doc_reg_names = {}
        self.filename_to_node = {}
        
        self.initialized = True
        try:
            self._load_all_data()
        except Exception as e:
            log.error(f"CitationGraph: Failed to load data: {e}")

    def _parse_filename(self, filename):
        parts = filename.replace(".txt", "").replace(".md", "").split("_")
        if len(parts) < 2: return None, None
        target_part = parts[-1] 
        article_match = re.search(r'(제\d+조)', target_part)
        appendix_match = re.search(r'(별표\s*제?\d+호?)', filename)
        reg_name = ""
        suffix = ""
        if article_match:
            suffix = article_match.group(1)
            reg_name_parts = parts[1:-1]
            reg_name = "_".join(reg_name_parts) if reg_name_parts else parts[0]
        elif appendix_match:
            suffix = re.sub(r'\s+', ' ', appendix_match.group(1))
            pre_match_str = filename.split(appendix_match.group(1))[0]
            pre_parts = pre_match_str.replace(".txt","").split('_')
            reg_name = "_".join(pre_parts[1:]).strip('_') if len(pre_parts) > 1 else pre_parts[0]
        else: return None, None
        return re.sub(r'\([^)]*\)', '', reg_name).strip(), suffix

    def _load_files(self, directory, doc_type):
        if not os.path.exists(directory):
            return
        
        # 본문은 .txt, 별표는 .md 로 보관되어 있어 둘 다 읽는다
        files = (glob.glob(os.path.join(directory, "*.txt"))
                 + glob.glob(os.path.join(directory, "*.md")))
        for file_path in files:
            file_name = os.path.basename(file_path)
            content = ""
            try:
                with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
            except:
                try:
                    with open(file_path, 'r', encoding='cp949') as f: content = f.read()
                except: continue
            
            reg_name, suffix = self._parse_filename(file_name)
            if reg_name and suffix:
                node_id = f"{reg_name}_{suffix}"
                self.documents[node_id] = content
                self.doc_titles[node_id] = file_name
                self.doc_reg_names[node_id] = reg_name 
                self.filename_to_node[file_name] = node_id
                self.graph.add_node(node_id, type=doc_type)

    def _load_all_data(self):
        log.info("CitationGraph: Loading data...")
        self._load_files(self.reg_dir, "regulation")
        self._load_files(self.app_dir, "appendix")
        
        for node_id, content in self.documents.items():
            current_reg_name = self.doc_reg_names.get(node_id)
            if not current_reg_name: continue
            
            reg_refs = re.findall(r'제\s*(\d+)\s*조', content)
            for ref_num in reg_refs:
                target_id = f"{current_reg_name}_제{ref_num}조"
                if target_id in self.documents and target_id != node_id:
                    self.graph.add_edge(node_id, target_id)
            
            app_refs = re.findall(r'별표\s*제?\s*(\d+)\s*호?', content)
            for ref_num in app_refs:
                target_id_1 = f"{current_reg_name}_별표 제{ref_num}호"
                target_id_2 = f"{current_reg_name}_별표 {ref_num}"
                target_id = target_id_1 if target_id_1 in self.documents else target_id_2 if target_id_2 in self.documents else None
                if target_id: 
                    self.graph.add_edge(node_id, target_id)
                    
        log.info(f"CitationGraph: Loaded {len(self.documents)} nodes.")

    def _is_appendix(self, node_id: str) -> bool:
        if "별표" in node_id:
            return True
        try:
            return self.graph.nodes[node_id].get("type") == "appendix"
        except Exception:
            return False

    def get_related_documents(self, retrieved_metadatas: list[dict], limit: int = 3) -> list[dict]:
        """
        벡터 검색 결과를 시드로 삼아, 인용 관계로 연결된 문서를 확장한다.

        양방향으로 탐색한다.
          - successors(정방향)  : 본문 조문 -> 그 조문이 인용하는 별표
          - predecessors(역방향): 별표 -> 그 별표를 인용하는 본문 조문

        [2026-08 측정] 평가셋 394건 중 본문+별표 연계형 74건 기준
            별표 검색 성공 78.4%  >  본문 검색 성공 64.9%
            즉 병목은 '별표를 못 찾는 것'이 아니라 '본문 조문을 못 찾는 것'이었다.
            회수 가능 건수: 역방향 13건(17.6%) vs 정방향 3건(4.1%)
        따라서 '별표 시드 -> 본문' 역방향 확장에 최우선 순위를 둔다.
        """
        seeds = []
        for meta in retrieved_metadatas:
            name = meta.get("name") or meta.get("source") or meta.get("title")
            if name and name in self.filename_to_node:
                seeds.append(self.filename_to_node[name])

        if not seeds:
            return []

        # (우선순위, 노드ID) — 숫자가 작을수록 먼저
        PRIO_APPENDIX_TO_REG = 0   # 별표 -> 본문 (역방향, 실측 회수량 최다)
        PRIO_REG_TO_APPENDIX = 1   # 본문 -> 별표 (정방향)
        PRIO_SAME_KIND = 2         # 조문 -> 조문 등 그 외

        candidates = []
        seen = set(seeds)

        for seed in seeds:
            seed_is_appendix = self._is_appendix(seed)
            try:
                # 역방향: 이 문서를 인용하고 있는 문서들
                for node in self.graph.predecessors(seed):
                    if node in seen:
                        continue
                    seen.add(node)
                    prio = (PRIO_APPENDIX_TO_REG
                            if seed_is_appendix and not self._is_appendix(node)
                            else PRIO_SAME_KIND)
                    candidates.append((prio, node))

                # 정방향: 이 문서가 인용하는 문서들
                for node in self.graph.successors(seed):
                    if node in seen:
                        continue
                    seen.add(node)
                    prio = (PRIO_REG_TO_APPENDIX
                            if not seed_is_appendix and self._is_appendix(node)
                            else PRIO_SAME_KIND)
                    candidates.append((prio, node))
            except Exception:
                continue

        candidates.sort(key=lambda x: x[0])
        final_nodes = [node_id for _, node_id in candidates[:limit]]
        
        results = []
        for node_id in final_nodes:
            content = self.documents.get(node_id)
            if content:
                title = self.doc_titles.get(node_id, node_id)
                results.append({
                    "page_content": content,
                    "metadata": {
                        "name": title,
                        "source": title,
                        "file_id": f"graph_{node_id}",
                        "description": "GraphRAG Expansion",
                    }
                })
                
        return results
