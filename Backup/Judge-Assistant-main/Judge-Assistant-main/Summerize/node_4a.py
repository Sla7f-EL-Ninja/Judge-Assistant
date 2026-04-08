import sys
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import uuid

# Add parent directory to path for shared schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import (
    LegalRoleEnum,
    AgreedBullet, DisputedPoint, PartyBullet,
    ThemeCluster, ThemedRole, Node4AOutput,
)


# --- Internal LLM Schemas ---

class ThemeAssignmentLLM(BaseModel):
    """LLM output: theme assignments for a batch of items."""
    theme_name: str = Field(description="اسم الموضوع الفرعي")
    item_ids: List[str] = Field(description="معرفات العناصر المنتمية لهذا الموضوع")


class ClusteringResultLLM(BaseModel):
    """LLM output: all theme assignments for one role."""
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


# --- Prompt Templates ---

SYSTEM_PROMPT_4A = """أنت مساعد قضائي متخصص في تنظيم المعلومات القانونية.

مهمتك: تجميع العناصر القانونية التالية المصنفة تحت دور "{role}" إلى مواضيع فرعية منطقية.

المواضيع المقترحة (يمكنك استخدامها أو إضافة مواضيع جديدة حسب المحتوى):
{suggested_themes}

القواعد:
1. أنشئ من 3 إلى 7 مواضيع فرعية
2. كل عنصر يجب أن ينتمي لموضوع واحد فقط
3. اختر أسماء مواضيع وصفية وواضحة بالعربية
4. لا تغير النصوص الأصلية - فقط صنف المعرفات
5. إذا كان عنصر لا يناسب أي موضوع مقترح، أنشئ موضوعاً جديداً مناسباً
6. لا تترك أي معرف بدون تصنيف"""

HUMAN_TEMPLATE_4A = """العناصر التالية مصنفة تحت دور "{role}":

{formatted_items}

جمّع هذه العناصر في مواضيع فرعية منطقية."""


class Node4A_ThematicClustering:
    MAX_ITEMS_PER_CALL = 50
    MIN_ITEMS_FOR_CLUSTERING = 6

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(ClusteringResultLLM)

    def assign_item_ids(self, role_agg: dict) -> Tuple[Dict[str, dict], List[Tuple[str, str]]]:
        """Assign temp IDs to all items in a RoleAggregation.

        Returns:
            id_lookup: {temp_id: original_item_dict}
            items_with_ids: [(temp_id, formatted_text)]
        """
        id_lookup: Dict[str, dict] = {}
        items_with_ids: List[Tuple[str, str]] = []

        # Agreed items
        for i, item in enumerate(role_agg.get("agreed", []), 1):
            temp_id = f"agreed-{i:03d}-{uuid.uuid4().hex[:8]}"
            id_lookup[temp_id] = {"type": "agreed", "data": item}
            sources_str = ", ".join(item.get("sources", []))
            text = f"[{temp_id}] [متفق عليه] {item.get('text', '')} [المصادر: {sources_str}]"
            items_with_ids.append((temp_id, text))

        # Disputed items
        for i, item in enumerate(role_agg.get("disputed", []), 1):
            temp_id = f"disputed-{i:03d}-{uuid.uuid4().hex[:8]}"
            id_lookup[temp_id] = {"type": "disputed", "data": item}
            # Format disputed item with subject and position summaries
            positions_text = []
            for pos in item.get("positions", []):
                party = pos.get("party", "")
                bullets_text = "; ".join(pos.get("bullets", []))
                sources_str = ", ".join(pos.get("sources", []))
                positions_text.append(f"{party}: {bullets_text} [المصادر: {sources_str}]")
            pos_summary = " | ".join(positions_text)
            text = f"[{temp_id}] [محل نزاع: {item.get('subject', '')}] {pos_summary}"
            items_with_ids.append((temp_id, text))

        # Party-specific items
        for i, item in enumerate(role_agg.get("party_specific", []), 1):
            temp_id = f"party-{i:03d}-{uuid.uuid4().hex[:8]}"
            id_lookup[temp_id] = {"type": "party_specific", "data": item}
            sources_str = ", ".join(item.get("sources", []))
            text = f"[{temp_id}] [{item.get('party', '')}] {item.get('text', '')} [المصادر: {sources_str}]"
            items_with_ids.append((temp_id, text))

        return id_lookup, items_with_ids

    def format_items_for_prompt(self, items_with_ids: List[Tuple[str, str]]) -> str:
        """Format items as text lines with IDs for the LLM."""
        return "\n".join(text for _, text in items_with_ids)

    def create_prompt_messages(self, formatted_items: str, role: str) -> list:
        """Build system + human messages."""
        suggested = ROLE_THEME_SUGGESTIONS.get(role, ["موضوع عام"])
        suggested_text = "\n".join(f"- {t}" for t in suggested)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_4A),
            ("human", HUMAN_TEMPLATE_4A),
        ])
        return prompt.format_messages(
            role=role,
            suggested_themes=suggested_text,
            formatted_items=formatted_items,
        )

    def cluster_batch(self, formatted_items: str, role: str) -> ClusteringResultLLM:
        """Single LLM call for one batch."""
        messages = self.create_prompt_messages(formatted_items, role)
        return self.parser.invoke(messages)

    def merge_batch_results(self, batch_results: List[ClusteringResultLLM]) -> Dict[str, List[str]]:
        """Merge theme assignments across batches by theme name."""
        merged: Dict[str, List[str]] = defaultdict(list)
        for result in batch_results:
            for theme in result.themes:
                merged[theme.theme_name].extend(theme.item_ids)
        return dict(merged)

    def validate_coverage(self, merged: Dict[str, List[str]], all_ids: set) -> Dict[str, List[str]]:
        """Ensure every ID is assigned to exactly one theme.

        - Missing IDs get assigned to fallback theme 'أخرى'
        - Duplicate IDs keep first occurrence only
        """
        seen: set = set()
        cleaned: Dict[str, List[str]] = {}

        for theme_name, item_ids in merged.items():
            unique_ids = []
            for item_id in item_ids:
                if item_id in seen:
                    print(f"Warning: item '{item_id}' duplicated across themes, keeping first occurrence.")
                    continue
                if item_id not in all_ids:
                    print(f"Warning: LLM returned unknown item_id '{item_id}', dropping.")
                    continue
                seen.add(item_id)
                unique_ids.append(item_id)
            if unique_ids:
                cleaned[theme_name] = unique_ids

        # Check for missing IDs
        missing = all_ids - seen
        if missing:
            print(f"Warning: {len(missing)} item(s) missing from LLM output, adding to 'أخرى' theme.")
            cleaned.setdefault("أخرى", []).extend(sorted(missing))

        # Theme count warning
        theme_count = len(cleaned)
        if theme_count < 2 or theme_count > 10:
            print(f"Warning: unusual theme count ({theme_count}), proceeding anyway.")

        return cleaned

    def reconstruct_themed_role(
        self, role: str, merged: Dict[str, List[str]], id_lookup: Dict[str, dict]
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
                item_type = entry["type"]
                data = entry["data"]

                if item_type == "agreed":
                    agreed.append(data)
                elif item_type == "disputed":
                    disputed.append(data)
                elif item_type == "party_specific":
                    party_specific.append(data)

            bullet_count = len(agreed) + len(disputed) + len(party_specific)
            themes.append({
                "theme_name": theme_name,
                "agreed": agreed,
                "disputed": disputed,
                "party_specific": party_specific,
                "bullet_count": bullet_count,
            })

        return {
            "role": role,
            "themes": themes,
        }

    def process_role(self, role_agg: dict) -> dict:
        """Process one RoleAggregation into a ThemedRole."""
        role = role_agg.get("role", "غير محدد")
        id_lookup, items_with_ids = self.assign_item_ids(role_agg)
        all_ids = set(id_lookup.keys())
        total_items = len(all_ids)

        print(f"  Role '{role}': {total_items} total items")

        # Small-role optimization: skip clustering if too few items
        if total_items < self.MIN_ITEMS_FOR_CLUSTERING:
            print(f"  Skipping clustering (< {self.MIN_ITEMS_FOR_CLUSTERING} items), single theme.")
            single_theme = {role: list(all_ids)}
            return self.reconstruct_themed_role(role, single_theme, id_lookup)

        try:
            # Decide single call vs batching
            if total_items <= self.MAX_ITEMS_PER_CALL:
                formatted = self.format_items_for_prompt(items_with_ids)
                result = self.cluster_batch(formatted, role)
                merged = self.merge_batch_results([result])
            else:
                # Split into batches
                batch_results = []
                for start in range(0, total_items, self.MAX_ITEMS_PER_CALL):
                    batch = items_with_ids[start:start + self.MAX_ITEMS_PER_CALL]
                    formatted = self.format_items_for_prompt(batch)
                    print(f"  Processing batch {start // self.MAX_ITEMS_PER_CALL + 1} "
                          f"({len(batch)} items)...")
                    result = self.cluster_batch(formatted, role)
                    batch_results.append(result)
                merged = self.merge_batch_results(batch_results)

            # Validate coverage
            merged = self.validate_coverage(merged, all_ids)

            return self.reconstruct_themed_role(role, merged, id_lookup)

        except Exception as e:
            print(f"  Error in LLM call for role '{role}': {e}")
            # Fallback: single theme = role name
            fallback = {role: list(all_ids)}
            return self.reconstruct_themed_role(role, fallback, id_lookup)

    def process(self, inputs: dict) -> dict:
        """Entry point.
        Input: Node3Output dict with {"role_aggregations": [...]}
        Output: Node4AOutput dict with {"themed_roles": [...]}
        """
        role_aggregations = inputs.get("role_aggregations", [])
        if not role_aggregations:
            return {"themed_roles": []}

        print("\n--- Node 4A: Thematic Clustering ---")
        themed_roles = []
        for role_agg in role_aggregations:
            themed_role = self.process_role(role_agg)
            themed_roles.append(themed_role)

        return {"themed_roles": themed_roles}
