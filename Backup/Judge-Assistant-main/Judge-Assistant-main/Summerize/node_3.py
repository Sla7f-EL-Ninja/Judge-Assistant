import sys
import os
from typing import List, Dict, Any
from collections import defaultdict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Add parent directory to path for shared schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import (
    LegalRoleEnum, PartyEnum,
    AgreedBullet, DisputePosition, DisputedPoint,
    PartyBullet, RoleAggregation, Node3Output,
)


# --- LLM Response Schemas (internal to Node 3) ---

class AgreedItemLLM(BaseModel):
    """An agreed-upon or uncontested point (LLM output)."""
    text: str = Field(description="النص الموحد للنقطة المتفق عليها")
    bullet_ids: List[str] = Field(description="معرفات النقاط الأصلية التي تدعم هذه النقطة")


class DisputeSideLLM(BaseModel):
    """One party's side in a dispute (LLM output)."""
    party: str = Field(description="اسم الطرف")
    bullet_ids: List[str] = Field(description="معرفات نقاط هذا الطرف")


class DisputedItemLLM(BaseModel):
    """A point of contention between parties (LLM output)."""
    subject: str = Field(description="موضوع النزاع باختصار")
    sides: List[DisputeSideLLM] = Field(description="موقف كل طرف")


class PartySpecificItemLLM(BaseModel):
    """A point unique to one party, not contested or matched (LLM output)."""
    party: str = Field(description="الطرف صاحب النقطة")
    bullet_ids: List[str] = Field(description="معرفات النقاط - قد تكون مدمجة من تكرارات")
    text: str = Field(description="النص الموحد بعد دمج التكرارات")


class RoleAggregationLLM(BaseModel):
    """Complete LLM output for one role."""
    agreed: List[AgreedItemLLM] = Field(description="نقاط متفق عليها أو غير متنازع عليها")
    disputed: List[DisputedItemLLM] = Field(description="نقاط محل نزاع بين الأطراف")
    party_specific: List[PartySpecificItemLLM] = Field(description="نقاط خاصة بطرف واحد")


# --- Prompt Templates ---

SYSTEM_PROMPT = """أنت مساعد قضائي متخصص في تحليل النزاعات القانونية المصرية.

مهمتك: تحليل مجموعة من النقاط القانونية المصنفة تحت دور "{role}" وتوزيعها على ثلاث فئات:

1. المتفق عليه: وقائع أو معلومات يقرها الطرفان صراحة، أو يذكرها أحدهما دون أن ينازع فيها الآخر.
2. محل النزاع: نقاط يتناقض فيها الأطراف مباشرة حول نفس الموضوع.
3. خاص بطرف: ادعاءات أو حجج أو طلبات تخص طرفاً واحداً ولا تقابلها نقطة من الطرف الآخر.

القواعد:
- كل نقطة (bullet_id) يجب أن تظهر في فئة واحدة فقط
- عند دمج نقاط مكررة من نفس الطرف، اذكر جميع معرفاتها
- في "محل النزاع"، حدد موضوع النزاع باختصار واذكر معرفات نقاط كل طرف
- في "المتفق عليه"، اكتب نصاً موحداً يعبر عن النقطة المتفق عليها
- الوقائع غير المتنازع عليها تعتبر "متفق عليه" حتى لو ذكرها طرف واحد فقط
- الادعاءات والحجج القانونية الخاصة بطرف واحد تصنف "خاص بطرف"
- استخدم اللغة العربية القانونية الرسمية
- لا تضف معلومات غير موجودة في النقاط الأصلية"""

HUMAN_TEMPLATE = """النقاط التالية مصنفة تحت دور "{role}":

{formatted_bullets}

حلل هذه النقاط ووزعها على الفئات الثلاث."""


class Node3_Aggregator:
    MAX_BULLETS_PER_CALL = 50

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(RoleAggregationLLM)

    def build_bullet_lookup(self, bullets: List[dict]) -> dict:
        """Returns {bullet_id: bullet_dict} for source resolution."""
        return {b["bullet_id"]: b for b in bullets}

    def group_by_role(self, bullets: List[dict]) -> dict:
        """Returns {role: [bullet_dicts]} using defaultdict."""
        groups = defaultdict(list)
        for b in bullets:
            groups[b["role"]].append(b)
        return groups

    def has_multiple_parties(self, bullets: List[dict]) -> bool:
        """Check if bullets come from more than one party."""
        parties = {b["party"] for b in bullets}
        return len(parties) > 1

    def format_bullets_for_prompt(self, bullets: List[dict]) -> str:
        """Format as: [bullet_id | party] bullet_text"""
        lines = []
        for b in bullets:
            lines.append(f"[{b['bullet_id']} | {b['party']}] {b['bullet']}")
        return "\n".join(lines)

    def create_prompt_messages(self, formatted_bullets: str, role: str) -> list:
        """Build system + human messages for the LLM."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_TEMPLATE),
        ])
        return prompt.format_messages(
            role=role,
            formatted_bullets=formatted_bullets,
        )

    def resolve_sources(self, bullet_ids: List[str], lookup: dict) -> List[str]:
        """Merge source lists from all referenced bullet_ids, deduped."""
        sources = []
        seen = set()
        for bid in bullet_ids:
            if bid not in lookup:
                continue
            for src in lookup[bid].get("source", []):
                if src not in seen:
                    seen.add(src)
                    sources.append(src)
        return sources

    def resolve_bullet_texts(self, bullet_ids: List[str], lookup: dict) -> List[str]:
        """Get exact original bullet texts for given IDs."""
        texts = []
        for bid in bullet_ids:
            if bid in lookup:
                texts.append(lookup[bid]["bullet"])
        return texts

    def validate_coverage(
        self,
        llm_result: RoleAggregationLLM,
        input_bullet_ids: set,
        bullets: List[dict],
    ) -> RoleAggregationLLM:
        """Ensure every input bullet_id appears in exactly one bucket.
        Missing IDs get added to party_specific.
        Duplicate IDs (in multiple buckets) keep first occurrence only."""

        # Build a map of bullet_id -> which bucket it first appeared in
        seen_ids: Dict[str, str] = {}

        # --- Agreed ---
        for item in llm_result.agreed:
            clean = []
            for bid in item.bullet_ids:
                if bid not in input_bullet_ids:
                    print(f"Warning: LLM returned unknown bullet_id '{bid}' in agreed, dropping.")
                    continue
                if bid in seen_ids:
                    print(f"Warning: bullet_id '{bid}' duplicated (first in {seen_ids[bid]}), skipping in agreed.")
                    continue
                seen_ids[bid] = "agreed"
                clean.append(bid)
            item.bullet_ids = clean

        # --- Disputed ---
        for item in llm_result.disputed:
            for side in item.sides:
                clean = []
                for bid in side.bullet_ids:
                    if bid not in input_bullet_ids:
                        print(f"Warning: LLM returned unknown bullet_id '{bid}' in disputed, dropping.")
                        continue
                    if bid in seen_ids:
                        print(f"Warning: bullet_id '{bid}' duplicated (first in {seen_ids[bid]}), skipping in disputed.")
                        continue
                    seen_ids[bid] = "disputed"
                    clean.append(bid)
                side.bullet_ids = clean

        # --- Party-specific ---
        for item in llm_result.party_specific:
            clean = []
            for bid in item.bullet_ids:
                if bid not in input_bullet_ids:
                    print(f"Warning: LLM returned unknown bullet_id '{bid}' in party_specific, dropping.")
                    continue
                if bid in seen_ids:
                    print(f"Warning: bullet_id '{bid}' duplicated (first in {seen_ids[bid]}), skipping in party_specific.")
                    continue
                seen_ids[bid] = "party_specific"
                clean.append(bid)
            item.bullet_ids = clean

        # --- Find missing IDs and add them to party_specific ---
        missing_ids = input_bullet_ids - set(seen_ids.keys())
        if missing_ids:
            bullet_map = {b["bullet_id"]: b for b in bullets}
            for mid in missing_ids:
                if mid not in bullet_map:
                    continue
                b = bullet_map[mid]
                print(f"Warning: bullet_id '{mid}' missing from LLM output, adding to party_specific.")
                llm_result.party_specific.append(
                    PartySpecificItemLLM(
                        party=b["party"],
                        bullet_ids=[mid],
                        text=b["bullet"],
                    )
                )

        return llm_result

    def build_role_aggregation(self, role: str, llm_result: RoleAggregationLLM, lookup: dict) -> dict:
        """Convert LLM result + lookup into final RoleAggregation dict."""

        # Agreed items
        agreed = []
        for item in llm_result.agreed:
            if not item.bullet_ids:
                continue
            agreed.append({
                "text": item.text,
                "sources": self.resolve_sources(item.bullet_ids, lookup),
            })

        # Disputed items
        disputed = []
        for item in llm_result.disputed:
            positions = []
            for side in item.sides:
                if not side.bullet_ids:
                    continue
                positions.append({
                    "party": side.party,
                    "bullets": self.resolve_bullet_texts(side.bullet_ids, lookup),
                    "sources": self.resolve_sources(side.bullet_ids, lookup),
                })
            if positions:
                disputed.append({
                    "subject": item.subject,
                    "positions": positions,
                })

        # Party-specific items
        party_specific = []
        for item in llm_result.party_specific:
            if not item.bullet_ids:
                continue
            party_specific.append({
                "party": item.party,
                "text": item.text,
                "sources": self.resolve_sources(item.bullet_ids, lookup),
            })

        return {
            "role": role,
            "agreed": agreed,
            "disputed": disputed,
            "party_specific": party_specific,
        }

    def process_role(self, role: str, bullets: List[dict], lookup: dict) -> dict:
        """Process all bullets for one role. Returns RoleAggregation dict."""

        # Single-party shortcut: no comparison possible
        if not self.has_multiple_parties(bullets):
            return {
                "role": role,
                "agreed": [],
                "disputed": [],
                "party_specific": [
                    {
                        "party": b["party"],
                        "text": b["bullet"],
                        "sources": b["source"],
                    }
                    for b in bullets
                ],
            }

        # Multi-party: call LLM
        input_bullet_ids = {b["bullet_id"] for b in bullets}
        formatted = self.format_bullets_for_prompt(bullets)

        try:
            messages = self.create_prompt_messages(formatted, role)
            llm_result = self.parser.invoke(messages)

            # Validate coverage
            llm_result = self.validate_coverage(llm_result, input_bullet_ids, bullets)

            # Build final output
            return self.build_role_aggregation(role, llm_result, lookup)

        except Exception as e:
            print(f"Error in LLM call for role '{role}': {e}")
            # Fallback: all bullets go to party_specific
            return {
                "role": role,
                "agreed": [],
                "disputed": [],
                "party_specific": [
                    {
                        "party": b["party"],
                        "text": b["bullet"],
                        "sources": b["source"],
                    }
                    for b in bullets
                ],
            }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point.
        Input: {"bullets": [...]} from Node 2
        Output: {"role_aggregations": [...]}"""

        bullets = inputs.get("bullets", [])
        if not bullets:
            return {"role_aggregations": []}

        lookup = self.build_bullet_lookup(bullets)
        role_groups = self.group_by_role(bullets)

        role_aggregations = []
        for role, role_bullets in role_groups.items():
            agg = self.process_role(role, role_bullets, lookup)
            role_aggregations.append(agg)

        return {"role_aggregations": role_aggregations}
