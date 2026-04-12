import uuid
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import LegalRoleEnum, PartyEnum, LegalBullet, Node2Output
from utils import get_logger, llm_invoke_with_retry

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


# --- Role-Specific Extraction Hints ---

ROLE_HINTS = {
    "الوقائع": "ركز على: الأحداث، التواريخ، الأفعال، العلاقات بين الأطراف، تسلسل الوقائع.",
    "الطلبات": "ركز على: ما يطلبه الخصم من المحكمة تحديداً، كل طلب منفصل في نقطة.",
    "الدفوع": "ركز على: كل دفع قانوني أو إجرائي منفصل، أسباب الدفع، النتيجة المطلوبة.",
    "المستندات": "ركز على: وصف كل مستند، ما يثبته، الطرف الذي قدّمه.",
    "الأساس القانوني": "ركز على: رقم المادة، اسم القانون، مبدأ النقض، وجه الانطباق.",
    "الإجراءات": "ركز على: تاريخ كل إجراء، نوعه، قرار المحكمة فيه.",
    "غير محدد": "حاول استخراج أي محتوى قانوني مفيد. إذا كان النص إدارياً بحتاً، أرجع قائمة فارغة.",
}

# --- Static system prompt template (no user content) ---
_SYSTEM_PROMPT_TEMPLATE = """أنت مساعد قضائي متخصص في استخراج النقاط القانونية من المستندات القضائية المصرية.

مهمتك: تحويل كل فقرة إلى نقاط قانونية ذرية (فكرة واحدة لكل نقطة).

التصنيف الحالي لهذه الفقرات: {role}
{role_hint}

القواعد الصارمة:
1. كل نقطة تحتوي على فكرة قانونية واحدة فقط
2. استخدم اللغة العربية القانونية الرسمية
3. لا تضف أي معلومات غير موجودة في النص الأصلي
4. لا تحذف أي فكرة جوهرية من النص
5. إذا كانت الفقرة قصيرة جداً أو لا تحتوي على محتوى قانوني، أرجع قائمة فارغة
6. حافظ على المصطلحات القانونية كما هي دون تبسيط
7. لا تغير المبالغ المالية أو الأرقام أو التواريخ — انقلها كما هي بالنص

لكل فقرة (محددة بمعرف chunk_id)، أرجع قائمة النقاط المستخرجة."""


class Node2_BulletExtractor:
    BATCH_SIZE = 5

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(BatchBulletResult)

    def build_citation(self, chunk: dict) -> str:
        """Build a human-readable source reference from chunk metadata."""
        return f"{chunk['doc_id']} ص{chunk['page_number']} ف{chunk['paragraph_number']}"

    def _build_messages(self, chunks: List[dict], role: str) -> list:
        """Build messages directly without ChatPromptTemplate (S2-4 safety)."""
        role_hint = ROLE_HINTS.get(role, ROLE_HINTS["غير محدد"])

        system_content = _SYSTEM_PROMPT_TEMPLATE.format(role=role, role_hint=role_hint)

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
        """Process a single batch: call LLM and return extracted bullet dicts."""
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

            # Fallback for chunk_ids the LLM missed
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
            # Fallback: wrap each chunk's clean_text as a single bullet
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
        Entry point for Node 2.
        Input:  {"classified_chunks": [ClassifiedChunk dicts]}
        Output: {"bullets": [LegalBullet dicts]}
        """
        classified_chunks = inputs.get("classified_chunks", [])
        if not classified_chunks:
            return {"bullets": []}

        # Filter out chunks with empty clean_text
        classified_chunks = [
            c for c in classified_chunks if c.get("clean_text", "").strip()
        ]
        if not classified_chunks:
            return {"bullets": []}

        # Group chunks by role
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
