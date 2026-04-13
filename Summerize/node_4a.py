import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import (
    LegalRoleEnum,
    AgreedBullet, DisputedPoint, PartyBullet,
    ThemeCluster, ThemedRole, Node4AOutput,
)
from utils import get_logger, llm_invoke_with_retry

logger = get_logger("hakim.node_4a")


# --- Internal LLM Schemas ---

class ThemeAssignmentLLM(BaseModel):
    """LLM output: theme assignment for one thematic group."""
    theme_name: str = Field(description="اسم الموضوع الفرعي")
    item_ids: List[str] = Field(description="معرفات العناصر المنتمية لهذا الموضوع")


class ClusteringResultLLM(BaseModel):
    """LLM output: all theme assignments for one role (or one batch)."""
    themes: List[ThemeAssignmentLLM] = Field(
        description="قائمة المواضيع الفرعية مع معرفات العناصر"
    )


# --- Predefined Theme Suggestions ---

ROLE_THEME_SUGGESTIONS: Dict[str, List[str]] = {
    "الوقائع": [
        "الوقائع التعاقدية",
        "الوقائع المالية",
        "الوقائع الإجرائية",
        "الخط الزمني للأحداث",
    ],
    "الطلبات": [
        "الطلبات الأصلية",
        "الطلبات الاحتياطية",
        "الطلبات الإجرائية",
    ],
    "الدفوع": [
        "دفوع شكلية",
        "دفوع موضوعية",
        "دفوع بالتقادم",
        "دفوع بعدم القبول",
    ],
    "المستندات": [
        "مستندات تعاقدية",
        "مستندات مالية",
        "مراسلات",
        "مستندات رسمية",
    ],
    "الأساس القانوني": [
        "قوانين مدنية",
        "قوانين إجرائية",
        "أحكام نقض",
        "مبادئ قانونية",
    ],
    "الإجراءات": [
        "إجراءات سابقة أمام نفس المحكمة",
        "إجراءات أمام محاكم أخرى",
        "إجراءات تنفيذية",
    ],
}

# --- Static prompt fragments ---

_SYSTEM_TEMPLATE = """أنت مساعد قضائي متخصص في تنظيم المعلومات القانونية.

مهمتك: تجميع العناصر القانونية التالية المصنفة تحت دور "{role}" إلى مواضيع فرعية منطقية.

المواضيع المقترحة (يمكنك استخدامها أو إضافة مواضيع جديدة حسب المحتوى):
{suggested_themes}

القواعد:
1. أنشئ من 3 إلى 7 مواضيع فرعية
2. كل عنصر يجب أن ينتمي لموضوع واحد فقط
3. اختر أسماء مواضيع وصفية وواضحة بالعربية
4. لا تغير النصوص الأصلية - فقط صنف المعرفات
5. إذا كان عنصر لا يناسب أي موضوع مقترح، أنشئ موضوعاً جديداً مناسباً
6. لا تترك أي معرف بدون تصنيف
7. يجب أن يكون كل موضوع مختلفاً جوهرياً عن بقية المواضيع — لا تنشئ موضوعين يتداخلان في المحتوى"""


class Node4A_ThematicClustering:
    MAX_ITEMS_PER_CALL = 50
    MIN_ITEMS_FOR_CLUSTERING = 6

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(ClusteringResultLLM)

    # --- Item ID assignment ---

    def assign_item_ids(
        self, role_agg: dict
    ) -> Tuple[Dict[str, dict], List[Tuple[str, str]]]:
        """Assign temp IDs to all items in a RoleAggregation.

        Returns:
            id_lookup:      {temp_id: {"type": ..., "data": ...}}
            items_with_ids: [(temp_id, formatted_text)]
        """
        id_lookup: Dict[str, dict] = {}
        items_with_ids: List[Tuple[str, str]] = []

        for i, item in enumerate(role_agg.get("agreed", []), 1):
            temp_id = f"agreed-{i:03d}-{uuid.uuid4().hex[:8]}"
            id_lookup[temp_id] = {"type": "agreed", "data": item}
            sources_str = ", ".join(item.get("sources", []))
            text = f"[{temp_id}] [متفق عليه] {item.get('text', '')} [المصادر: {sources_str}]"
            items_with_ids.append((temp_id, text))

        for i, item in enumerate(role_agg.get("disputed", []), 1):
            temp_id = f"disputed-{i:03d}-{uuid.uuid4().hex[:8]}"
            id_lookup[temp_id] = {"type": "disputed", "data": item}
            positions_text = []
            for pos in item.get("positions", []):
                party = pos.get("party", "")
                bullets_text = "; ".join(pos.get("bullets", []))
                sources_str = ", ".join(pos.get("sources", []))
                positions_text.append(f"{party}: {bullets_text} [المصادر: {sources_str}]")
            pos_summary = " | ".join(positions_text)
            text = f"[{temp_id}] [محل نزاع: {item.get('subject', '')}] {pos_summary}"
            items_with_ids.append((temp_id, text))

        for i, item in enumerate(role_agg.get("party_specific", []), 1):
            temp_id = f"party-{i:03d}-{uuid.uuid4().hex[:8]}"
            id_lookup[temp_id] = {"type": "party_specific", "data": item}
            sources_str = ", ".join(item.get("sources", []))
            text = (
                f"[{temp_id}] [{item.get('party', '')}] "
                f"{item.get('text', '')} [المصادر: {sources_str}]"
            )
            items_with_ids.append((temp_id, text))

        return id_lookup, items_with_ids

    def format_items_for_prompt(self, items_with_ids: List[Tuple[str, str]]) -> str:
        """Format items as text lines with IDs for the LLM."""
        return "\n".join(text for _, text in items_with_ids)

    # --- Message construction (S2-4: direct messages, not ChatPromptTemplate) ---

    def _build_messages(
        self,
        formatted_items: str,
        role: str,
        existing_theme_names: List[str] = None,
    ) -> list:
        """Build system + human messages directly.

        Args:
            formatted_items:     Pre-formatted item text block.
            role:                Legal role name.
            existing_theme_names: S2-8 — theme names discovered in previous
                                  batches; included in suggested themes so
                                  the LLM reuses them instead of inventing
                                  divergent names.
        """
        suggested = list(ROLE_THEME_SUGGESTIONS.get(role, ["موضوع عام"]))

        # S2-8: Prepend already-discovered theme names so the LLM prefers them
        if existing_theme_names:
            # Put existing names first so they appear most prominent
            combined = existing_theme_names + [s for s in suggested if s not in existing_theme_names]
            suggested_text = "\n".join(f"- {t}" for t in combined)
            continuity_note = (
                "\nملاحظة: المواضيع المُبدوءة بـ ★ مُستخدمة مسبقاً — "
                "استخدمها إن أمكن للحفاظ على التوحيد عبر المجموعات."
            )
            suggested_text = (
                "\n".join(
                    f"- ★ {t}" if t in existing_theme_names else f"- {t}"
                    for t in combined
                )
                + continuity_note
            )
        else:
            suggested_text = "\n".join(f"- {t}" for t in suggested)

        system_content = _SYSTEM_TEMPLATE.format(
            role=role, suggested_themes=suggested_text
        )
        human_content = (
            f'العناصر التالية مصنفة تحت دور "{role}":\n\n'
            f"{formatted_items}\n\n"
            "جمّع هذه العناصر في مواضيع فرعية منطقية."
        )
        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    # --- LLM clustering ---

    def cluster_batch(
        self,
        formatted_items: str,
        role: str,
        existing_theme_names: List[str] = None,
    ) -> ClusteringResultLLM:
        """Single LLM call for one batch."""
        messages = self._build_messages(formatted_items, role, existing_theme_names)
        return llm_invoke_with_retry(self.parser, messages, logger=logger)

    def merge_batch_results(
        self, batch_results: List[ClusteringResultLLM]
    ) -> Dict[str, List[str]]:
        """Merge theme assignments across batches by theme name (exact match)."""
        merged: Dict[str, List[str]] = defaultdict(list)
        for result in batch_results:
            for theme in result.themes:
                merged[theme.theme_name].extend(theme.item_ids)
        return dict(merged)

    # --- Coverage validation ---

    def validate_coverage(
        self, merged: Dict[str, List[str]], all_ids: set
    ) -> Dict[str, List[str]]:
        """Ensure every ID is assigned to exactly one theme.

        Missing IDs → 'أخرى' fallback theme.
        Duplicate IDs → keep first occurrence.
        """
        seen: set = set()
        cleaned: Dict[str, List[str]] = {}

        for theme_name, item_ids in merged.items():
            unique_ids = []
            for item_id in item_ids:
                if item_id in seen:
                    logger.warning("Item '%s' duplicated across themes, keeping first.", item_id)
                    continue
                if item_id not in all_ids:
                    logger.warning("Unknown item_id '%s' from LLM, dropping.", item_id)
                    continue
                seen.add(item_id)
                unique_ids.append(item_id)
            if unique_ids:
                cleaned[theme_name] = unique_ids

        missing = all_ids - seen
        if missing:
            logger.warning("%d item(s) missing from LLM output, adding to 'أخرى'.", len(missing))
            cleaned.setdefault("أخرى", []).extend(sorted(missing))

        theme_count = len(cleaned)
        if theme_count < 2 or theme_count > 10:
            logger.warning("Unusual theme count (%d), proceeding anyway.", theme_count)

        return cleaned

    # --- Reconstruction ---

    def reconstruct_themed_role(
        self,
        role: str,
        merged: Dict[str, List[str]],
        id_lookup: Dict[str, dict],
    ) -> dict:
        """Rebuild ThemeCluster objects from merged assignments + lookup."""
        themes = []
        for theme_name, item_ids in merged.items():
            agreed = []
            disputed = []
            party_specific = []

            for item_id in item_ids:
                if item_id not in id_lookup:
                    continue
                entry = id_lookup[item_id]
                if entry["type"] == "agreed":
                    agreed.append(entry["data"])
                elif entry["type"] == "disputed":
                    disputed.append(entry["data"])
                elif entry["type"] == "party_specific":
                    party_specific.append(entry["data"])

            themes.append({
                "theme_name": theme_name,
                "agreed": agreed,
                "disputed": disputed,
                "party_specific": party_specific,
                "bullet_count": len(agreed) + len(disputed) + len(party_specific),
            })

        return {"role": role, "themes": themes}

    # --- Main processing ---

    def process_role(self, role_agg: dict) -> dict:
        """Process one RoleAggregation into a ThemedRole."""
        role = role_agg.get("role", "غير محدد")
        id_lookup, items_with_ids = self.assign_item_ids(role_agg)
        all_ids = set(id_lookup.keys())
        total_items = len(all_ids)

        logger.info("  Role '%s': %d total items", role, total_items)

        # Small-role optimization: skip clustering
        if total_items < self.MIN_ITEMS_FOR_CLUSTERING:
            logger.info(
                "  Skipping clustering (< %d items), single theme.", self.MIN_ITEMS_FOR_CLUSTERING
            )
            return self.reconstruct_themed_role(role, {role: list(all_ids)}, id_lookup)

        try:
            if total_items <= self.MAX_ITEMS_PER_CALL:
                # Single call
                formatted = self.format_items_for_prompt(items_with_ids)
                result = self.cluster_batch(formatted, role)
                merged = self.merge_batch_results([result])
            else:
                # S2-8: Multi-batch — pass prior theme names to subsequent batches
                batch_results: List[ClusteringResultLLM] = []
                discovered_themes: List[str] = []

                for start in range(0, total_items, self.MAX_ITEMS_PER_CALL):
                    batch = items_with_ids[start : start + self.MAX_ITEMS_PER_CALL]
                    formatted = self.format_items_for_prompt(batch)
                    batch_num = start // self.MAX_ITEMS_PER_CALL + 1
                    logger.info(
                        "  Processing batch %d (%d items)...", batch_num, len(batch)
                    )

                    # Pass discovered theme names to keep naming consistent
                    result = self.cluster_batch(
                        formatted, role,
                        existing_theme_names=discovered_themes if discovered_themes else None,
                    )
                    batch_results.append(result)

                    # Update discovered themes for the next batch
                    for theme in result.themes:
                        if theme.theme_name not in discovered_themes:
                            discovered_themes.append(theme.theme_name)

                merged = self.merge_batch_results(batch_results)

            merged = self.validate_coverage(merged, all_ids)
            return self.reconstruct_themed_role(role, merged, id_lookup)

        except Exception as e:
            logger.error("Error in LLM call for role '%s': %s", role, e)
            fallback = {role: list(all_ids)}
            return self.reconstruct_themed_role(role, fallback, id_lookup)

    def process(self, inputs: dict) -> dict:
        """
        Input:  {"role_aggregations": [RoleAggregation dicts]}
        Output: {"themed_roles": [ThemedRole dicts]}
        """
        role_aggregations = inputs.get("role_aggregations", [])
        if not role_aggregations:
            return {"themed_roles": []}

        logger.info("--- Node 4A: Thematic Clustering ---")
        max_workers = min(len(role_aggregations), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            themed_roles = list(ex.map(self.process_role, role_aggregations))
        return {"themed_roles": themed_roles}
