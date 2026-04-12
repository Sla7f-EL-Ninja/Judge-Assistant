from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import LegalRoleEnum
from utils import get_logger, llm_invoke_with_retry

logger = get_logger("hakim.node_1")


# --- LLM Response Schemas ---

class ClassificationItem(BaseModel):
    chunk_id: str = Field(description="The ID of the chunk")
    role: LegalRoleEnum = Field(description="The legal role of the chunk")


class BatchClassificationResult(BaseModel):
    classifications: List[ClassificationItem] = Field(
        description="List of classified items"
    )


# --- System prompt (static, no user content) ---
_SYSTEM_PROMPT = """أنت مساعد قضائي ذكي. مهمتك تصنيف الفقرات القانونية إلى واحدة من الفئات التالية بدقة:
[الوقائع, الطلبات, الدفوع, المستندات, الأساس القانوني, الإجراءات, غير محدد]

تعليمات:
1. "الوقائع": السرد القصصي وتاريخ النزاع.
2. "الطلبات": ما يطلبه الخصم من المحكمة في الختام.
3. "الدفوع": الردود القانونية، الدفع بعدم الاختصاص، التقادم، إلخ.
4. "المستندات": الإشارة للمرفقات والأدلة الكتابية.
5. "الأساس القانوني": نصوص المواد وأحكام النقض.
6. "الإجراءات": سير الدعوى والجلسات السابقة.
7. "غير محدد": فقرات إدارية بحتة أو لا تنتمي لأي فئة أعلاه.

صنف كل فقرة بناءً على محتواها وسياق المستند."""


class Node1_RoleClassifier:
    BATCH_SIZE = 10

    def __init__(self, llm):
        self.llm = llm
        self.parser = self.llm.with_structured_output(BatchClassificationResult)

    def _build_messages(self, chunks: List[dict], doc_meta: Dict[str, Any]) -> list:
        """Build system + human messages without ChatPromptTemplate.

        Constructing messages directly avoids any risk of curly braces in
        chunk text being misinterpreted as template variables (S2-4).
        """
        doc_type = doc_meta.get("doc_type", "غير محدد")
        party = doc_meta.get("party", "غير محدد")

        formatted_text = ""
        for chunk in chunks:
            formatted_text += (
                f"ID: {chunk.get('chunk_id')}\n"
                f"Text: {chunk.get('clean_text')}\n---\n"
            )

        system_content = (
            f"معلومات المستند:\n- النوع: {doc_type}\n- مقدم من: {party}\n\n"
            + _SYSTEM_PROMPT
        )
        human_content = formatted_text

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def process_batch(
        self, chunks: List[dict], doc_meta: Dict[str, Any]
    ) -> List[dict]:
        """Classify one batch. Returns NEW dicts (does not mutate inputs).

        S2-9: Create new dicts with {**chunk, role, confidence} instead of
        mutating the input chunk dicts in-place.
        """
        try:
            messages = self._build_messages(chunks, doc_meta)
            result = llm_invoke_with_retry(self.parser, messages, logger=logger)

            role_map = {item.chunk_id: item.role for item in result.classifications}

            return [
                {**chunk, "role": role_map.get(chunk.get("chunk_id"), "غير محدد"), "confidence": 1.0}
                for chunk in chunks
            ]

        except Exception as e:
            logger.error("Error in batch classification: %s", e)
            # Fallback: mark all chunks in this batch as unclassified
            return [{**chunk, "role": "غير محدد", "confidence": 0.0} for chunk in chunks]

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:  {"chunks": [NormalizedChunk dicts]}
        Output: {"classified_chunks": [ClassifiedChunk dicts]}
        """
        all_chunks = inputs.get("chunks", [])
        if not all_chunks:
            return {"classified_chunks": []}

        # S2-2: Group chunks by (doc_type, party) so each group is processed
        # with its own correct metadata in the system prompt, rather than
        # using the first chunk's metadata for the entire input.
        doc_groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for chunk in all_chunks:
            key = (
                chunk.get("doc_type", "غير محدد"),
                chunk.get("party", "غير محدد"),
            )
            doc_groups[key].append(chunk)

        classified_chunks: List[dict] = []

        def _classify_group(args):
            (doc_type, party), group_chunks = args
            doc_meta = {"doc_type": doc_type, "party": party}
            logger.info(
                "  Classifying %d chunk(s) for doc_type='%s', party='%s'",
                len(group_chunks), doc_type, party,
            )
            results: List[dict] = []
            for i in range(0, len(group_chunks), self.BATCH_SIZE):
                batch = group_chunks[i : i + self.BATCH_SIZE]
                results.extend(self.process_batch(batch, doc_meta))
            return results

        group_items = list(doc_groups.items())
        max_workers = min(len(group_items), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for result in ex.map(_classify_group, group_items):
                classified_chunks.extend(result)

        return {"classified_chunks": classified_chunks}
