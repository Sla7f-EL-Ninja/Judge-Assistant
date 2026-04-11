import hashlib
import re
import uuid
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import DocumentMetadata, NormalizedChunk, DocTypeEnum, PartyEnum
from utils import get_logger, normalize_arabic_for_matching

logger = get_logger("hakim.node_0")

# --- Configuration ---
PAGE_SIZE_ESTIMATE = 2000  # Characters per page if no markers

# --- Heuristics ---
DOC_TYPE_KEYWORDS = {
    "صحيفة دعوى": ["صحيفة افتتاح", "عريضة دعوى", "طلب افتتاح", "صحيفة دعوى"],
    "مذكرة دفاع": ["مذكرة بدفاع", "مذكرة دفاع", "مذكرة رد"],
    "حافظة مستندات": ["حافظة مستندات", "بيان مستندات"],
    "محضر جلسة": ["محضر جلسة"],
    "حكم تمهيدي": ["حكم تمهيدي"],
}

PARTY_KEYWORDS = {
    "المدعي": ["مقدمة من / ... (المدعي)", "المدعي", "الطالب"],
    "المدعى عليه": ["المدعى عليه", "المعلن إليه"],
    "النيابة": ["النيابة العامة"],
    "المحكمة": ["المحكمة", "الهيئة الموقرة"],
    "خبير": ["تقرير خبير", "الخبير"],
}

# S2-10: Pre-normalize keyword lists for robust matching (Hamza / Alef Maksura variants)
_DOC_TYPE_KEYWORDS_NORM = {
    dtype: [normalize_arabic_for_matching(k) for k in kws]
    for dtype, kws in DOC_TYPE_KEYWORDS.items()
}
_PARTY_KEYWORDS_NORM = {
    party: [normalize_arabic_for_matching(k) for k in kws]
    for party, kws in PARTY_KEYWORDS.items()
}


# --- Node 0 Class ---
class Node0_DocumentIntake:
    def __init__(self, llm):
        self.llm = llm
        self.metadata_parser = llm.with_structured_output(DocumentMetadata)

    def clean_text(self, text: str) -> str:
        # 1. Remove Unicode directional marks
        text = text.replace("\u200f", "").replace("\u200e", "")
        # 2. Remove Tatweel (Kashida)
        text = re.sub(r"[ـ]+", "", text)
        # 3. Remove page numbers like  - 12 -
        text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
        # 4. S2-7: Remove header lines more precisely — only match two consecutive
        #    lines where one mentions وزارة العدل and the next mentions محكمة.
        #    Using MULTILINE + anchored pattern avoids eating legitimate paragraphs.
        text = re.sub(
            r"^[^\n]*وزارة العدل[^\n]*$\n^[^\n]*محكمة[^\n]*$",
            "",
            text,
            flags=re.MULTILINE,
        )
        # 5. Remove certification stamps
        text = re.sub(r"صورة طبق الأصل", "", text)
        # 6. Collapse horizontal whitespace only
        text = re.sub(r"[ \t]+", " ", text).strip()
        return text

    def extract_metadata(self, header_text: str) -> DocumentMetadata:
        """Try regex heuristic first, fall back to LLM."""
        found_type: DocTypeEnum = "غير محدد"
        found_party: PartyEnum = "غير محدد"

        # S2-10: Normalize header for matching only — preserves original for LLM
        norm_header = normalize_arabic_for_matching(header_text)

        for dtype, keywords in _DOC_TYPE_KEYWORDS_NORM.items():
            if any(k in norm_header for k in keywords):
                found_type = dtype
                break

        for party, keywords in _PARTY_KEYWORDS_NORM.items():
            if any(k in norm_header for k in keywords):
                found_party = party
                break

        # If both found by heuristic, skip LLM call
        if found_type != "غير محدد" and found_party != "غير محدد":
            return DocumentMetadata(doc_type=found_type, party=found_party)

        # LLM Fallback
        prompt = ChatPromptTemplate.from_messages([
            ("system", """أنت نظام ذكي لتصنيف المستندات القانونية المصرية.
مهمتك: استخراج (نوع المستند) و (الصفة القانونية للجهة المقدمة) من النص التالي.

القواعد:
1. نوع المستند (doc_type) يجب أن يكون واحداً من: ["صحيفة دعوى", "مذكرة دفاع", "مذكرة رد", "حافظة مستندات", "محضر جلسة", "حكم تمهيدي", "غير محدد"].
2. الطرف (party) يجب أن يكون واحداً من: ["المدعي", "المدعى عليه", "النيابة", "المحكمة", "خبير", "غير محدد"].
3. إذا كان النص غير واضح، ارجع "غير محدد"."""),
            ("human", "{text}")
        ])

        try:
            return self.metadata_parser.invoke(prompt.format(text=header_text[:2000]))
        except Exception as e:
            logger.warning("LLM metadata extraction failed: %s", e)
            return DocumentMetadata(doc_type="غير محدد", party="غير محدد")

    def segment_document(
        self, clean_text: str, doc_id: str, metadata: DocumentMetadata
    ) -> List[dict]:
        """Split text into chunks with page/paragraph tracking."""
        chunks = []
        raw_paragraphs = clean_text.split("\n\n")

        current_page = 1
        char_count = 0

        # S2-5: Include a content hash in the UUID seed so that two documents
        # with the same doc_id (e.g. both named "unknown") still produce unique
        # chunk_ids for their respective paragraphs.
        content_hash = hashlib.md5(
            clean_text[:500].encode("utf-8", errors="replace")
        ).hexdigest()[:8]

        for idx, para in enumerate(raw_paragraphs):
            para = para.strip()
            if not para:
                continue

            char_count += len(para)
            if char_count > PAGE_SIZE_ESTIMATE:
                current_page += 1
                char_count = 0

            chunk = NormalizedChunk(
                chunk_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        f"{doc_id}_{content_hash}_{idx + 1}",
                    )
                ),
                doc_id=doc_id,
                page_number=current_page,
                paragraph_number=idx + 1,
                clean_text=para,
                party=metadata.party,
                doc_type=metadata.doc_type,
            )
            # S1-9: Use model_dump() (Pydantic v2); .dict() is deprecated
            chunks.append(chunk.model_dump())

        return chunks

    def process(self, inputs: dict) -> dict:
        """Main entry point.
        Input:  {"raw_text": "...", "doc_id": "..."}
        Output: {"chunks": [NormalizedChunk dicts]}
        """
        raw_text = inputs["raw_text"]
        doc_id = inputs.get("doc_id", "unknown")

        header_sample = raw_text[:2000]
        metadata = self.extract_metadata(header_sample)
        logger.info(
            "  doc_id='%s'  doc_type='%s'  party='%s'",
            doc_id, metadata.doc_type, metadata.party,
        )

        clean_body = self.clean_text(raw_text)
        chunks = self.segment_document(clean_body, doc_id, metadata)

        return {"chunks": chunks}
