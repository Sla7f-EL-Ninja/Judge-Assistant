import re
import uuid
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import DocumentMetadata, NormalizedChunk, DocTypeEnum, PartyEnum

# --- Configuration ---
# Assuming LLM is passed from outside, but model name kept for reference
PAGE_SIZE_ESTIMATE = 2000 # Characters per page if no markers

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
    "خبير": ["تقرير خبير", "الخبير"]
}


# --- Node 0 Class ---
class Node0_DocumentIntake:
    def __init__(self, llm):
        self.llm = llm
        self.metadata_parser = llm.with_structured_output(DocumentMetadata)
        
    def clean_text(self, text: str) -> str:
        # 1. Normalize Unicode
        text = text.replace("\u200f", "").replace("\u200e", "") # Remove directional marks
        # 2. Remove Tatweel (Kashida)
        text = re.sub(r"[ـ]+", "", text)
        # 3. Remove Page Numbers (e.g., - 12 -)
        text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
        # 4. Remove Header Garbage (e.g., Ministry of Justice...) - Simple regex for now
        text = re.sub(r"وزارة العدل.*محكمة.*", "", text)
        # 5. Remove Stamps
        text = re.sub(r"صورة طبق الأصل", "", text)
        # We only collapse horizontal whitespace.
        text = re.sub(r"[ \t]+", " ", text).strip()


        return text

    def extract_metadata(self, header_text: str) -> DocumentMetadata:
        """
        Tries regex first, falls back to LLM.
        """
        # 1. Regex Heuristic
        found_type: DocTypeEnum = "غير محدد"
        found_party: PartyEnum = "غير محدد"
        
        for dtype, keywords in DOC_TYPE_KEYWORDS.items():
            if any(k in header_text for k in keywords):
                found_type = dtype
                break
        
        for party, keywords in PARTY_KEYWORDS.items():
             if any(k in header_text for k in keywords):
                found_party = party
                break

        # If confident (both found), return simple object
        if found_type != "غير محدد" and found_party != "غير محدد":
             return DocumentMetadata(doc_type=found_type, party=found_party)
        
        # 2. LLM Fallback
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
            print(f"LLM Metadata extraction failed: {e}")
            return DocumentMetadata(doc_type="غير محدد", party="غير محدد")

    def segment_document(self, clean_text: str, doc_id: str, metadata: DocumentMetadata) -> List[dict]:
        """
        Splits text into chunks with page/para tracking.
        """
        chunks = []
        # Naive split by double newline for paragraphs
        raw_paragraphs = clean_text.split('\n\n') 
        
        current_page = 1
        char_count = 0
        
        for idx, para in enumerate(raw_paragraphs):
            para = para.strip()
            if not para: continue
            
            # Merge small chunks logic (simplified from plan: < 50 chars merge)
            # For now, we will just keep them as is or implement simple merge if needed.
            # Implementing simple merge for very short paragraphs might be good but let's stick to base plan
            
            # Update page estimation
            char_count += len(para)
            if char_count > PAGE_SIZE_ESTIMATE:
                current_page += 1
                char_count = 0
            
            chunk = NormalizedChunk(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{idx + 1}")),
                doc_id=doc_id,
                page_number=current_page, # Or extracted from explicit markers if available
                paragraph_number=idx + 1,
                clean_text=para,
                party=metadata.party,
                doc_type=metadata.doc_type
            )
            chunks.append(chunk.dict())
            
        return chunks

    def process(self, inputs: dict) -> dict:
        """
        Main entry point for LangGraph node.
        Input: {"raw_text": "...", "doc_id": "..."}
        """
        raw_text = inputs["raw_text"]
        doc_id = inputs.get("doc_id", "unknown")
        
        # 1. Identify Metadata (from first 2000 chars)
        header_sample = raw_text[:2000]
        metadata = self.extract_metadata(header_sample)
        
        # 2. Deep Clean
        clean_body = self.clean_text(raw_text)
        
        # 3. Segment
        chunks = self.segment_document(clean_body, doc_id, metadata)
        
        return {"chunks": chunks}
