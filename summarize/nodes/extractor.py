import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from summarize.schemas import LegalRoleEnum, PartyEnum, LegalBullet, Node2Output
from summarize.utils import get_logger, llm_invoke_with_retry
from summarize.prompts.extractor import SYSTEM_PROMPT_TEMPLATE, ROLE_HINTS

logger = get_logger("hakim.node_2")


# --- LLM Response Schemas (internal to Node 2) ---

class ChunkBullets(BaseModel):
    """LLM output: extracted bullets for a single chunk."""
    chunk_id: str = Field(description="معرف الفقرة الأصلية")
    bullets: List[str] = Field(
        description="قائمة النقاط القانونية المستخرجة - فكرة واحدة لكل نقطة"
    )


class BatchBulletResult(BaseModel):
    """LLM output: bullets for all chunks in a batch."""
    extractions: List[ChunkBullets]


class Node2_BulletExtractor:
    BATCH_SIZE = 3

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(BatchBulletResult)

    def build_citation(self, chunk: dict) -> str:
        return f"{chunk['doc_id']} ص{chunk['page_number']} ف{chunk['paragraph_number']}"

    def _build_messages(self, chunks: List[dict], role: str) -> list:
        role_hint = ROLE_HINTS.get(role, ROLE_HINTS["غير محدد"])

        system_content = SYSTEM_PROMPT_TEMPLATE.format(role=role, role_hint=role_hint)

        formatted_chunks = ""
        for chunk in chunks:
            formatted_chunks += (
                f"ID: {chunk['chunk_id']}\n"
                f"النص: {chunk['clean_text']}\n---\n"
            )

        human_content = (
            f'الفقرات التالية مصنفة كـ "{role}". '
            f"استخرج النقاط القانونية الذرية من كل فقرة:\n\n{formatted_chunks}"
        )

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def process_batch(self, chunks: List[dict], role: str) -> List[dict]:
        chunk_map = {c["chunk_id"]: c for c in chunks}
        results = []

        try:
            messages = self._build_messages(chunks, role)
            batch_result = llm_invoke_with_retry(self.parser, messages, logger=logger)

            seen_ids = set()

            for extraction in batch_result.extractions:
                cid = extraction.chunk_id

                if cid not in chunk_map:
                    logger.warning("LLM returned unknown chunk_id '%s', dropping.", cid)
                    continue

                seen_ids.add(cid)
                source_chunk = chunk_map[cid]
                citation = self.build_citation(source_chunk)

                for bullet_text in extraction.bullets:
                    bullet_text = bullet_text.strip()
                    if not bullet_text:
                        continue
                    results.append({
                        "bullet_id": str(uuid.uuid4()),
                        "role": source_chunk.get("role", role),
                        "bullet": bullet_text,
                        "source": [citation],
                        "party": source_chunk.get("party", "غير محدد"),
                        "chunk_id": cid,
                    })

            for cid, chunk in chunk_map.items():
                if cid not in seen_ids:
                    clean_text = chunk.get("clean_text", "").strip()
                    if not clean_text:
                        continue
                    logger.warning("LLM missed chunk_id '%s', using fallback.", cid)
                    results.append({
                        "bullet_id": str(uuid.uuid4()),
                        "role": chunk.get("role", role),
                        "bullet": clean_text,
                        "source": [self.build_citation(chunk)],
                        "party": chunk.get("party", "غير محدد"),
                        "chunk_id": cid,
                    })

        except Exception as e:
            logger.error("Error in batch bullet extraction: %s", e)
            for chunk in chunks:
                clean_text = chunk.get("clean_text", "").strip()
                if not clean_text:
                    continue
                results.append({
                    "bullet_id": str(uuid.uuid4()),
                    "role": chunk.get("role", role),
                    "bullet": clean_text,
                    "source": [self.build_citation(chunk)],
                    "party": chunk.get("party", "غير محدد"),
                    "chunk_id": chunk.get("chunk_id", ""),
                })

        return results

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:  {"classified_chunks": [ClassifiedChunk dicts]}
        Output: {"bullets": [LegalBullet dicts]}
        """
        classified_chunks = inputs.get("classified_chunks", [])
        if not classified_chunks:
            return {"bullets": []}

        classified_chunks = [
            c for c in classified_chunks if c.get("clean_text", "").strip()
        ]
        if not classified_chunks:
            return {"bullets": []}

        role_groups: Dict[str, List[dict]] = defaultdict(list)
        for chunk in classified_chunks:
            role_groups[chunk.get("role", "غير محدد")].append(chunk)

        all_bullets: List[dict] = []

        def _extract_role(args):
            role, chunks = args
            results: List[dict] = []
            for i in range(0, len(chunks), self.BATCH_SIZE):
                batch = chunks[i : i + self.BATCH_SIZE]
                results.extend(self.process_batch(batch, role))
            return results

        role_items = list(role_groups.items())
        max_workers = min(len(role_items), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for result in ex.map(_extract_role, role_items):
                all_bullets.extend(result)

        return {"bullets": all_bullets}
