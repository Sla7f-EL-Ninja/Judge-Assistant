from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path for shared schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import LegalRoleEnum

# Define the output structure for the LLM (Batch)
class ClassificationItem(BaseModel):
    chunk_id: str = Field(description="The ID of the chunk")
    role: LegalRoleEnum = Field(description="The legal role of the chunk")

class BatchClassificationResult(BaseModel):
    classifications: List[ClassificationItem] = Field(description="List of classified items")

class Node1_RoleClassifier:
    def __init__(self, llm):
        self.llm = llm
        self.parser = self.llm.with_structured_output(BatchClassificationResult)
    
    def create_prompt_messages(self, chunks: List[dict], doc_meta: Dict[str, Any]):
        # Format chunks for the prompt
        formatted_text = ""
        for chunk in chunks:
            c_id = chunk.get('chunk_id')
            c_text = chunk.get('clean_text')
            formatted_text += f"ID: {c_id}\nText: {c_text}\n---\n"
            
        system_prompt = """
        أنت مساعد قضائي ذكي. مهمتك تصنيف الفقرات القانونية إلى واحدة من الفئات التالية بدقة:
        [الوقائع, الطلبات, الدفوع, المستندات, الأساس القانوني, الإجراءات, غير محدد]

        معلومات المستند:
        - النوع: {doc_type}
        - مقدم من: {party}

        تعليمات:
        1. "الوقائع": السرد القصسي وتاريخ النزاع.
        2. "الطلبات": ما يطلبه الخصم من المحكمة في الختام.
        3. "الدفوع": الردود القانونية، الدفع بعدم الاختصاص، التقادم، إلخ.
        4. "المستندات": الإشارة للمرفقات والأدلة الكتابية.
        5. "الأساس القانوني": نصوص المواد وأحكام النقض.
        6. "الإجراءات": سير الدعوى والجلسات السابقة.

        صنف كل فقرة بناءً على محتواها وسياق المستند.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{formatted_batch_text}")
        ])
        
        # Manually format the template with input variables
        # Note: ChatPromptTemplate.format_messages returns a list of BaseMessage
        return prompt.format_messages(
            doc_type=doc_meta.get('doc_type', 'غير محدد'),
            party=doc_meta.get('party', 'غير محدد'),
            formatted_batch_text=formatted_text
        )

    def process_batch(self, chunks: List[dict], doc_meta: Dict[str, Any]) -> List[dict]:
        try:
            prompt_messages = self.create_prompt_messages(chunks, doc_meta)
            # Use invoke directly with messages
            result = self.parser.invoke(prompt_messages)
            
            # Merge results back into chunks
            role_map = {item.chunk_id: item.role for item in result.classifications}
            
            for chunk in chunks:
                c_id = chunk.get('chunk_id')
                # If ID not found in result, fallback to 'غير محدد'
                assigned_role = role_map.get(c_id, "غير محدد")
                chunk['role'] = assigned_role
                chunk['confidence'] = 1.0 
                
        except Exception as e:
            print(f"Error in batch classification: {e}")
            # Fallback for the whole batch
            for chunk in chunks:
                chunk['role'] = "غير محدد"
                chunk['confidence'] = 0.0
                
        return chunks

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: {"chunks": [NormalizedChunk dicts...]} (from Node 0)
        Output: {"classified_chunks": [ClassifiedChunk dicts...]}
        """
        all_chunks = inputs.get("chunks", [])
        classified_chunks = []
        
        if not all_chunks:
             return {"classified_chunks": []}

        # Batch size of 10
        BATCH_SIZE = 10
        
        # Determine metadata from the first chunk (assuming homogeneity for now)
        first_chunk = all_chunks[0]
        doc_type = first_chunk.get('doc_type', "غير محدد")
        party = first_chunk.get('party', "غير محدد")
        doc_meta = {"doc_type": doc_type, "party": party}

        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            
            # Process batch
            processed_batch = self.process_batch(batch, doc_meta)
            classified_chunks.extend(processed_batch)
            
        return {"classified_chunks": classified_chunks}
