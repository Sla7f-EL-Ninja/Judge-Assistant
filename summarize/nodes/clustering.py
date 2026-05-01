from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import uuid

from summarize.schemas import (
    LegalRoleEnum,
    AgreedBullet, DisputedPoint, PartyBullet,
    ThemeCluster, ThemedRole, Node4AOutput,
)
from summarize.utils import get_logger, llm_invoke_with_retry
from summarize.prompts.clustering import ROLE_THEME_SUGGESTIONS, SYSTEM_TEMPLATE

logger = get_logger("hakim.node_4a")


# --- Internal LLM Schemas ---

class ThemeAssignmentLLM(BaseModel):
    theme_name: str = Field(description="اسم الموضوع الفرعي")
    item_ids: List[str] = Field(description="معرفات العناصر المنتمية لهذا الموضوع")


class ClusteringResultLLM(BaseModel):
    themes: List[ThemeAssignmentLLM] = Field(
        description="قائمة المواضيع الفرعية مع معرفات العناصر"
    )


class Node4A_ThematicClustering:
    MAX_ITEMS_PER_CALL = 50
    MIN_ITEMS_FOR_CLUSTERING = 6

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(ClusteringResultLLM)

    def assign_item_ids(
        self, role_agg: dict
    ) -> Tuple[Dict[str, dict], List[Tuple[str, str]]]:
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
        return "\n".join(text for _, text in items_with_ids)

    def _build_messages(
        self,
        formatted_items: str,
        role: str,
        existing_theme_names: List[str] = None,
    ) -> list:
        suggested = list(ROLE_THEME_SUGGESTIONS.get(role, ["موضوع عام"]))

        if existing_theme_names:
            combined = existing_theme_names + [s for s in suggested if s not in existing_theme_names]
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

        system_content = SYSTEM_TEMPLATE.format(
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

    def cluster_batch(
        self,
        formatted_items: str,
        role: str,
        existing_theme_names: List[str] = None,
    ) -> ClusteringResultLLM:
        messages = self._build_messages(formatted_items, role, existing_theme_names)
        return llm_invoke_with_retry(self.parser, messages, logger=logger)

    def merge_batch_results(
        self, batch_results: List[ClusteringResultLLM]
    ) -> Dict[str, List[str]]:
        merged: Dict[str, List[str]] = defaultdict(list)
        for result in batch_results:
            for theme in result.themes:
                merged[theme.theme_name].extend(theme.item_ids)
        return dict(merged)

    def validate_coverage(
        self, merged: Dict[str, List[str]], all_ids: set
    ) -> Dict[str, List[str]]:
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

    def reconstruct_themed_role(
        self,
        role: str,
        merged: Dict[str, List[str]],
        id_lookup: Dict[str, dict],
    ) -> dict:
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

    def process_role(self, role_agg: dict) -> dict:
        role = role_agg.get("role", "غير محدد")
        id_lookup, items_with_ids = self.assign_item_ids(role_agg)
        all_ids = set(id_lookup.keys())
        total_items = len(all_ids)

        logger.info("  Role '%s': %d total items", role, total_items)

        if total_items < self.MIN_ITEMS_FOR_CLUSTERING:
            logger.info(
                "  Skipping clustering (< %d items), single theme.", self.MIN_ITEMS_FOR_CLUSTERING
            )
            return self.reconstruct_themed_role(role, {role: list(all_ids)}, id_lookup)

        try:
            if total_items <= self.MAX_ITEMS_PER_CALL:
                formatted = self.format_items_for_prompt(items_with_ids)
                result = self.cluster_batch(formatted, role)
                merged = self.merge_batch_results([result])
            else:
                batch_results: List[ClusteringResultLLM] = []
                discovered_themes: List[str] = []

                for start in range(0, total_items, self.MAX_ITEMS_PER_CALL):
                    batch = items_with_ids[start : start + self.MAX_ITEMS_PER_CALL]
                    formatted = self.format_items_for_prompt(batch)
                    batch_num = start // self.MAX_ITEMS_PER_CALL + 1
                    logger.info(
                        "  Processing batch %d (%d items)...", batch_num, len(batch)
                    )

                    result = self.cluster_batch(
                        formatted, role,
                        existing_theme_names=discovered_themes if discovered_themes else None,
                    )
                    batch_results.append(result)

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
