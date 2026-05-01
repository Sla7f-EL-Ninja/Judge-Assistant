import re
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from summarize.schemas import (
    LegalRoleEnum,
    ThemeSummary, RoleThemeSummaries, Node4BOutput,
)
from summarize.utils import get_logger, llm_invoke_with_retry
from summarize.prompts.synthesis import SYSTEM_TEMPLATE

logger = get_logger("hakim.node_4b")


# ---------------------------------------------------------------------------
# Internal LLM Schemas
# ---------------------------------------------------------------------------

class SentenceLLM(BaseModel):
    text: str = Field(description="الجملة القانونية — جملة واحدة فقط")
    source_items: List[str] = Field(
        description="معرّفات العناصر المُدخلة (A###، D###، P###) التي تستند إليها هذه الجملة — يجب ذكر معرّف واحد على الأقل"
    )


class SynthesisResultLLM(BaseModel):
    sentences: List[SentenceLLM] = Field(
        description="قائمة الجمل — جملة واحدة لكل فكرة مع معرّفات مصادرها"
    )
    key_disputes: List[str] = Field(
        description="عناوين مختصرة لنقاط الخلاف الجوهرية"
    )


class Node4B_ThemeSynthesis:

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(SynthesisResultLLM)

    def _assign_item_ids(
        self, theme_cluster: dict
    ) -> Tuple[Dict[str, str], str, str, str]:
        id_map: Dict[str, str] = {}

        agreed_lines = []
        for i, item in enumerate(theme_cluster.get("agreed", []), 1):
            iid = f"A{i:03d}"
            sources_str = ", ".join(item.get("sources", []))
            text = item.get("text", "")
            id_map[iid] = text
            agreed_lines.append(f"[{iid}] {text} [المصادر: {sources_str}]")

        disputed_lines = []
        for i, item in enumerate(theme_cluster.get("disputed", []), 1):
            iid = f"D{i:03d}"
            subject = item.get("subject", "")
            positions_parts = []
            for pos in item.get("positions", []):
                party = pos.get("party", "")
                bullets_text = "; ".join(pos.get("bullets", []))
                sources_str = ", ".join(pos.get("sources", []))
                positions_parts.append(
                    f"{party}: {bullets_text} [المصادر: {sources_str}]"
                )
            full_text = f"[محل نزاع: {subject}] " + " | ".join(positions_parts)
            id_map[iid] = subject
            disputed_lines.append(f"[{iid}] {full_text}")

        party_specific_lines = []
        for i, item in enumerate(theme_cluster.get("party_specific", []), 1):
            iid = f"P{i:03d}"
            party = item.get("party", "")
            text = item.get("text", "")
            sources_str = ", ".join(item.get("sources", []))
            id_map[iid] = text
            party_specific_lines.append(
                f"[{iid}] [{party}] {text} [المصادر: {sources_str}]"
            )

        agreed_text = "\n".join(agreed_lines) if agreed_lines else "لا يوجد"
        disputed_text = "\n".join(disputed_lines) if disputed_lines else "لا يوجد"
        party_specific_text = (
            "\n".join(party_specific_lines) if party_specific_lines else "لا يوجد"
        )
        return id_map, agreed_text, disputed_text, party_specific_text

    def collect_sources(self, theme_cluster: dict) -> List[str]:
        sources: List[str] = []
        seen: set = set()

        for item in theme_cluster.get("agreed", []):
            for s in item.get("sources", []):
                if s not in seen:
                    seen.add(s)
                    sources.append(s)

        for item in theme_cluster.get("disputed", []):
            for pos in item.get("positions", []):
                for s in pos.get("sources", []):
                    if s not in seen:
                        seen.add(s)
                        sources.append(s)

        for item in theme_cluster.get("party_specific", []):
            for s in item.get("sources", []):
                if s not in seen:
                    seen.add(s)
                    sources.append(s)

        return sources

    def _build_messages(
        self,
        theme: str,
        role: str,
        agreed_text: str,
        disputed_text: str,
        party_specific_text: str,
        all_item_ids: List[str],
        party_manifest: Optional[Dict] = None,
    ) -> list:
        party_absence_clause = ""
        if party_manifest:
            parties_without_defense = [
                p for p, doc_types in party_manifest.items()
                if "مذكرة دفاع" not in doc_types
                and "مذكرة رد" not in doc_types
                and p not in ("المحكمة", "غير محدد", "خبير")
            ]
            if parties_without_defense:
                names = "، ".join(parties_without_defense)
                party_absence_clause = (
                    f"- الأطراف الذين لم يقدموا مذكرات دفاع في الملف: {names}. "
                    f"إذا طُلب وصف موقف أي من هؤلاء، اكتب حرفياً: "
                    f"'لم ترد وثائق دفاعية لهذا الطرف'.\n"
                )

        system_content = SYSTEM_TEMPLATE.format(
            theme=theme,
            role=role,
            party_absence_clause=party_absence_clause,
        )

        all_ids_str = "، ".join(all_item_ids) if all_item_ids else "لا يوجد"

        human_content = (
            f'الموضوع: "{theme}" ضمن "{role}"\n\n'
            f"معرّفات العناصر المتاحة: {all_ids_str}\n\n"
            f"النقاط المتفق عليها:\n{agreed_text}\n\n"
            f"النقاط المتنازع عليها:\n{disputed_text}\n\n"
            f"النقاط الخاصة بكل طرف:\n{party_specific_text}\n\n"
            "اكتب الجمل مع الإشارة إلى معرّفات العناصر المصدرية لكل جملة."
        )
        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def build_fallback_summary(self, theme_cluster: dict) -> str:
        parts = []

        agreed = theme_cluster.get("agreed", [])
        if agreed:
            parts.append("النقاط المتفق عليها:")
            for item in agreed:
                parts.append(f"- {item.get('text', '')}")

        disputed = theme_cluster.get("disputed", [])
        if disputed:
            parts.append("النقاط المتنازع عليها:")
            for item in disputed:
                parts.append(f"- {item.get('subject', '')}")
                for pos in item.get("positions", []):
                    bullets_text = "; ".join(pos.get("bullets", []))
                    parts.append(f"  * {pos.get('party', '')}: {bullets_text}")

        party_specific = theme_cluster.get("party_specific", [])
        if party_specific:
            parts.append("النقاط الخاصة بكل طرف:")
            for item in party_specific:
                parts.append(f"- [{item.get('party', '')}] {item.get('text', '')}")

        return "[ملخص خام - يحتاج مراجعة]\n" + "\n".join(parts)

    def extract_dispute_subjects(self, disputed: list) -> List[str]:
        return [item.get("subject", "") for item in disputed if item.get("subject")]

    def synthesize_theme(
        self,
        theme_cluster: dict,
        role: str,
        party_manifest: Optional[Dict] = None,
    ) -> dict:
        theme_name = theme_cluster.get("theme_name", "")
        disputed = theme_cluster.get("disputed", [])
        sources = self.collect_sources(theme_cluster)

        id_map, agreed_text, disputed_text, party_specific_text = (
            self._assign_item_ids(theme_cluster)
        )
        all_item_ids = list(id_map.keys())

        try:
            messages = self._build_messages(
                theme_name, role, agreed_text, disputed_text, party_specific_text,
                all_item_ids, party_manifest,
            )
            llm_result = llm_invoke_with_retry(self.parser, messages, logger=logger)

            valid_sentences = [s for s in llm_result.sentences if s.source_items]
            orphan_count = len(llm_result.sentences) - len(valid_sentences)
            if orphan_count:
                logger.warning(
                    "Stripped %d unsourced sentence(s) from theme '%s'.",
                    orphan_count, theme_name,
                )

            if valid_sentences:
                summary = " ".join(s.text for s in valid_sentences)
            else:
                logger.warning(
                    "No valid sentences for theme '%s', using fallback.", theme_name
                )
                summary = self.build_fallback_summary(theme_cluster)

            key_disputes = llm_result.key_disputes
            if disputed and not key_disputes:
                key_disputes = self.extract_dispute_subjects(disputed)

            return {
                "theme": theme_name,
                "summary": summary,
                "key_disputes": key_disputes,
                "sources": sources,
            }

        except Exception as e:
            logger.error("Error in LLM call for theme '%s': %s", theme_name, e)
            return {
                "theme": theme_name,
                "summary": self.build_fallback_summary(theme_cluster),
                "key_disputes": self.extract_dispute_subjects(disputed),
                "sources": sources,
            }

    def process_role(
        self, themed_role: dict, party_manifest: Optional[Dict] = None
    ) -> dict:
        role = themed_role.get("role", "غير محدد")
        themes = themed_role.get("themes", [])

        logger.info("  Role '%s': %d theme(s) to synthesize", role, len(themes))

        if not themes:
            return {"role": role, "theme_summaries": []}

        results: List[Any] = [None] * len(themes)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(
                    self.synthesize_theme, theme_cluster, role, party_manifest
                ): i
                for i, theme_cluster in enumerate(themes)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                theme_cluster = themes[idx]
                theme_name = theme_cluster.get("theme_name", "")
                try:
                    results[idx] = future.result()
                    logger.info(
                        "    Theme '%s' (%d items) synthesized.",
                        theme_name, theme_cluster.get("bullet_count", 0),
                    )
                except Exception as e:
                    logger.error(
                        "Unexpected error for theme '%s': %s", theme_name, e
                    )
                    results[idx] = {
                        "theme": theme_name,
                        "summary": self.build_fallback_summary(theme_cluster),
                        "key_disputes": self.extract_dispute_subjects(
                            theme_cluster.get("disputed", [])
                        ),
                        "sources": self.collect_sources(theme_cluster),
                    }

        return {"role": role, "theme_summaries": [r for r in results if r is not None]}

    def _collect_input_numbers(self, themed_roles: list) -> set:
        _pat = re.compile(r'\b\d[\d,./٠-٩]*\b')
        numbers: set = set()
        for tr in themed_roles:
            for theme in tr.get("themes", []):
                for item in theme.get("agreed", []):
                    numbers.update(_pat.findall(item.get("text", "")))
                for item in theme.get("disputed", []):
                    for pos in item.get("positions", []):
                        for b in pos.get("bullets", []):
                            numbers.update(_pat.findall(b))
                for item in theme.get("party_specific", []):
                    numbers.update(_pat.findall(item.get("text", "")))
        return numbers

    def _check_numeric_fabrication(
        self, summaries: list, themed_roles: list
    ) -> None:
        source_numbers = self._collect_input_numbers(themed_roles)
        _pat = re.compile(r'\b\d[\d,./٠-٩]*\b')

        for role_summary in summaries:
            for ts in role_summary.get("theme_summaries", []):
                summary_text = ts.get("summary", "")
                summary_numbers = set(_pat.findall(summary_text))
                fabricated = {
                    n for n in (summary_numbers - source_numbers)
                    if len(n) > 1
                }
                if fabricated:
                    logger.warning(
                        "⚠️ Potential fabricated numbers in theme '%s': %s",
                        ts.get("theme", ""), sorted(fabricated),
                    )

    def process(self, inputs: dict) -> dict:
        """
        Input:  {"themed_roles": [...], "party_manifest": {...}}
        Output: {"role_theme_summaries": [...]}
        """
        themed_roles = inputs.get("themed_roles", [])
        party_manifest = inputs.get("party_manifest", {})

        if not themed_roles:
            return {"role_theme_summaries": []}

        logger.info(
            "--- Node 4B: Theme-Level Synthesis (%d role(s)) ---", len(themed_roles)
        )

        def _process_one_role(themed_role: dict) -> dict:
            return self.process_role(themed_role, party_manifest)

        max_workers = min(len(themed_roles), 6)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            role_theme_summaries = list(
                executor.map(_process_one_role, themed_roles)
            )

        self._check_numeric_fabrication(role_theme_summaries, themed_roles)

        return {"role_theme_summaries": role_theme_summaries}
