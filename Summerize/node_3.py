import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import (
    LegalRoleEnum, PartyEnum,
    AgreedBullet, DisputePosition, DisputedPoint,
    PartyBullet, RoleAggregation, Node3Output,
)
from utils import get_logger, llm_invoke_with_retry

logger = get_logger("hakim.node_3")


# --- LLM Response Schemas (internal to Node 3) ---

class AgreedItemLLM(BaseModel):
    """An agreed-upon or uncontested point (LLM output)."""
    text: str = Field(description="النص الموحد للنقطة المتفق عليها")
    bullet_ids: List[str] = Field(description="معرفات النقاط الأصلية التي تدعم هذه النقطة")


class DisputeSideLLM(BaseModel):
    """One party's side in a dispute (LLM output)."""
    # S2-6: Use PartyEnum to constrain to valid party values
    party: PartyEnum = Field(description="اسم الطرف")
    bullet_ids: List[str] = Field(description="معرفات نقاط هذا الطرف")


class DisputedItemLLM(BaseModel):
    """A point of contention between parties (LLM output)."""
    subject: str = Field(description="موضوع النزاع باختصار")
    sides: List[DisputeSideLLM] = Field(description="موقف كل طرف")


class PartySpecificItemLLM(BaseModel):
    """A point unique to one party, not contested or matched (LLM output)."""
    # S2-6: Use PartyEnum to constrain to valid party values
    party: PartyEnum = Field(description="الطرف صاحب النقطة")
    bullet_ids: List[str] = Field(description="معرفات النقاط - قد تكون مدمجة من تكرارات")
    text: str = Field(description="النص الموحد بعد دمج التكرارات")


class RoleAggregationLLM(BaseModel):
    """Complete LLM output for one role."""
    agreed: List[AgreedItemLLM] = Field(description="نقاط متفق عليها أو غير متنازع عليها")
    disputed: List[DisputedItemLLM] = Field(description="نقاط محل نزاع بين الأطراف")
    party_specific: List[PartySpecificItemLLM] = Field(description="نقاط خاصة بطرف واحد")


# --- Static prompt content (no user content) ---

_SYSTEM_PROMPT_TEMPLATE = """أنت مساعد قضائي متخصص في تحليل النزاعات القانونية المصرية.

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


class Node3_Aggregator:
    MAX_BULLETS_PER_CALL = 50

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(RoleAggregationLLM)

    # --- Helpers ---

    def build_bullet_lookup(self, bullets: List[dict]) -> dict:
        """Returns {bullet_id: bullet_dict} for source resolution."""
        return {b["bullet_id"]: b for b in bullets}

    def group_by_role(self, bullets: List[dict]) -> dict:
        """Returns {role: [bullet_dicts]}."""
        groups: Dict[str, List[dict]] = defaultdict(list)
        for b in bullets:
            groups[b["role"]].append(b)
        return groups

    def has_multiple_parties(self, bullets: List[dict]) -> bool:
        """Check if bullets come from more than one party."""
        return len({b["party"] for b in bullets}) > 1

    def format_bullets_for_prompt(self, bullets: List[dict]) -> str:
        """Format as: [bullet_id | party] bullet_text"""
        return "\n".join(
            f"[{b['bullet_id']} | {b['party']}] {b['bullet']}" for b in bullets
        )

    def _build_messages(self, formatted_bullets: str, role: str) -> list:
        """Build system + human messages directly (S2-4: avoids template escaping issues)."""
        system_content = _SYSTEM_PROMPT_TEMPLATE.format(role=role)
        human_content = (
            f'النقاط التالية مصنفة تحت دور "{role}":\n\n'
            f"{formatted_bullets}\n\n"
            "حلل هذه النقاط ووزعها على الفئات الثلاث."
        )
        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def resolve_sources(self, bullet_ids: List[str], lookup: dict) -> List[str]:
        """Merge source lists from all referenced bullet_ids, deduped."""
        sources: List[str] = []
        seen: set = set()
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
        return [lookup[bid]["bullet"] for bid in bullet_ids if bid in lookup]

    # --- Coverage validation ---

    def validate_coverage(
        self,
        llm_result: RoleAggregationLLM,
        input_bullet_ids: set,
        bullets: List[dict],
    ) -> RoleAggregationLLM:
        """Ensure every input bullet_id appears in exactly one bucket.

        Missing IDs → added to party_specific.
        Duplicate IDs → keep first occurrence only.
        """
        seen_ids: Dict[str, str] = {}

        for item in llm_result.agreed:
            clean = []
            for bid in item.bullet_ids:
                if bid not in input_bullet_ids:
                    logger.warning("Unknown bullet_id '%s' in agreed, dropping.", bid)
                    continue
                if bid in seen_ids:
                    logger.warning("Duplicate bullet_id '%s' (first in %s), skipping.", bid, seen_ids[bid])
                    continue
                seen_ids[bid] = "agreed"
                clean.append(bid)
            item.bullet_ids = clean

        for item in llm_result.disputed:
            for side in item.sides:
                clean = []
                for bid in side.bullet_ids:
                    if bid not in input_bullet_ids:
                        logger.warning("Unknown bullet_id '%s' in disputed, dropping.", bid)
                        continue
                    if bid in seen_ids:
                        logger.warning("Duplicate bullet_id '%s' (first in %s), skipping.", bid, seen_ids[bid])
                        continue
                    seen_ids[bid] = "disputed"
                    clean.append(bid)
                side.bullet_ids = clean

        for item in llm_result.party_specific:
            clean = []
            for bid in item.bullet_ids:
                if bid not in input_bullet_ids:
                    logger.warning("Unknown bullet_id '%s' in party_specific, dropping.", bid)
                    continue
                if bid in seen_ids:
                    logger.warning("Duplicate bullet_id '%s' (first in %s), skipping.", bid, seen_ids[bid])
                    continue
                seen_ids[bid] = "party_specific"
                clean.append(bid)
            item.bullet_ids = clean

        # Add missing IDs to party_specific
        missing_ids = input_bullet_ids - set(seen_ids.keys())
        if missing_ids:
            bullet_map = {b["bullet_id"]: b for b in bullets}
            for mid in missing_ids:
                if mid not in bullet_map:
                    continue
                b = bullet_map[mid]
                logger.warning("bullet_id '%s' missing from LLM output, adding to party_specific.", mid)
                llm_result.party_specific.append(
                    PartySpecificItemLLM(
                        party=b["party"],
                        bullet_ids=[mid],
                        text=b["bullet"],
                    )
                )

        return llm_result

    # --- Output construction ---

    def build_role_aggregation(
        self, role: str, llm_result: RoleAggregationLLM, lookup: dict
    ) -> dict:
        """Convert LLM result + lookup into final RoleAggregation dict."""

        agreed = [
            {
                "text": item.text,
                "sources": self.resolve_sources(item.bullet_ids, lookup),
            }
            for item in llm_result.agreed
            if item.bullet_ids
        ]

        disputed = []
        for item in llm_result.disputed:
            positions = []
            for side in item.sides:
                if not side.bullet_ids:
                    continue
                # Anchor party label to source-document ground truth rather
                # than trusting the LLM's inference (which can reverse parties).
                # If all bullets for this side share the same original party,
                # use that; otherwise fall back to the LLM's label.
                source_parties = {
                    lookup[bid]["party"]
                    for bid in side.bullet_ids
                    if bid in lookup
                }
                inferred_party = (
                    source_parties.pop() if len(source_parties) == 1 else side.party
                )
                positions.append(
                    {
                        "party": inferred_party,
                        "bullets": self.resolve_bullet_texts(side.bullet_ids, lookup),
                        "sources": self.resolve_sources(side.bullet_ids, lookup),
                    }
                )
            if positions:
                disputed.append({"subject": item.subject, "positions": positions})

        party_specific = [
            {
                "party": item.party,
                "text": item.text,
                "sources": self.resolve_sources(item.bullet_ids, lookup),
            }
            for item in llm_result.party_specific
            if item.bullet_ids
        ]

        return {
            "role": role,
            "agreed": agreed,
            "disputed": disputed,
            "party_specific": party_specific,
        }

    def _call_llm_for_batch(self, bullets: List[dict], role: str) -> RoleAggregationLLM:
        """Single LLM call for a subset of bullets."""
        formatted = self.format_bullets_for_prompt(bullets)
        messages = self._build_messages(formatted, role)
        return llm_invoke_with_retry(self.parser, messages, logger=logger)

    def _fallback_aggregation(self, bullets: List[dict]) -> RoleAggregationLLM:
        """Fallback: put every bullet into party_specific."""
        return RoleAggregationLLM(
            agreed=[],
            disputed=[],
            party_specific=[
                PartySpecificItemLLM(
                    party=b["party"],
                    bullet_ids=[b["bullet_id"]],
                    text=b["bullet"],
                )
                for b in bullets
            ],
        )

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

        input_bullet_ids = {b["bullet_id"] for b in bullets}

        # S2-3 / S1-4: Enforce MAX_BULLETS_PER_CALL — batch large roles
        if len(bullets) <= self.MAX_BULLETS_PER_CALL:
            # Single LLM call (common path)
            try:
                llm_result = self._call_llm_for_batch(bullets, role)
            except Exception as e:
                logger.error("LLM call failed for role '%s': %s", role, e)
                llm_result = self._fallback_aggregation(bullets)
        else:
            # Multiple batches — merge results afterwards
            logger.info(
                "  Role '%s': %d bullets exceeds MAX_BULLETS_PER_CALL=%d, batching.",
                role, len(bullets), self.MAX_BULLETS_PER_CALL,
            )
            batch_results: List[RoleAggregationLLM] = []
            for start in range(0, len(bullets), self.MAX_BULLETS_PER_CALL):
                batch = bullets[start : start + self.MAX_BULLETS_PER_CALL]
                logger.info(
                    "    Processing batch %d (%d bullets)...",
                    start // self.MAX_BULLETS_PER_CALL + 1, len(batch),
                )
                try:
                    batch_result = self._call_llm_for_batch(batch, role)
                except Exception as e:
                    logger.error("LLM call failed for role '%s' batch: %s", role, e)
                    batch_result = self._fallback_aggregation(batch)
                batch_results.append(batch_result)

            # Merge by simple concatenation; validate_coverage handles dedup & missing
            llm_result = RoleAggregationLLM(
                agreed=[a for r in batch_results for a in r.agreed],
                disputed=[d for r in batch_results for d in r.disputed],
                party_specific=[p for r in batch_results for p in r.party_specific],
            )

        # Validate and build final output
        llm_result = self.validate_coverage(llm_result, input_bullet_ids, bullets)
        return self.build_role_aggregation(role, llm_result, lookup)

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:  {"bullets": [LegalBullet dicts]}
        Output: {"role_aggregations": [RoleAggregation dicts]}
        """
        bullets = inputs.get("bullets", [])
        if not bullets:
            return {"role_aggregations": []}

        lookup = self.build_bullet_lookup(bullets)
        role_groups = self.group_by_role(bullets)

        def _process_role_item(args):
            role, role_bullets = args
            logger.info("  Processing role '%s' (%d bullets)", role, len(role_bullets))
            return self.process_role(role, role_bullets, lookup)

        role_items = list(role_groups.items())
        max_workers = min(len(role_items), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            role_aggregations = list(ex.map(_process_role_item, role_items))

        return {"role_aggregations": role_aggregations}
