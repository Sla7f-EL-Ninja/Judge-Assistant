import sys
import os
import concurrent.futures
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import (
    LegalRoleEnum,
    ThemeSummary, RoleThemeSummaries, Node4BOutput,
)
from utils import get_logger, llm_invoke_with_retry

logger = get_logger("hakim.node_4b")


# --- Internal LLM Schema ---

class SynthesisResultLLM(BaseModel):
    """LLM output: synthesis for one theme."""
    summary: str = Field(description="ملخص الموضوع في 2-3 فقرات")
    key_disputes: List[str] = Field(description="عناوين مختصرة لنقاط الخلاف الجوهرية")


# --- Static system prompt template (no user content) ---

_SYSTEM_TEMPLATE = """أنت مساعد قضائي متخصص في تلخيص المعلومات القانونية للقاضي.

مهمتك: كتابة ملخص في 2-3 فقرات لموضوع "{theme}" ضمن "{role}".

الملخص يجب أن يشمل:
1. النقاط المتفق عليها أو غير المتنازع عليها (إن وجدت)
2. نقاط الخلاف الجوهرية مع ذكر موقف كل خصم
3. النقاط الخاصة بكل طرف

شروط صارمة:
- استخدم اللغة العربية القانونية الرسمية
- استخدم صيغ المقارنة عند وجود خلاف: "يتمسك... بينما يدفع..."، "ينازع... ويستند إلى..."
- لا تضف أي رأي أو استنتاج أو توصية
- حافظ على المصطلحات القانونية كما هي
- لا تختلق وقائع غير موجودة في النقاط المقدمة
- اذكر عناوين مختصرة لنقاط الخلاف الجوهرية"""


class Node4B_ThemeSynthesis:

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(SynthesisResultLLM)

    # --- Formatting helpers ---

    def format_agreed(self, agreed: list) -> str:
        if not agreed:
            return "لا يوجد"
        lines = []
        for item in agreed:
            sources_str = ", ".join(item.get("sources", []))
            lines.append(f"- {item.get('text', '')} [المصادر: {sources_str}]")
        return "\n".join(lines)

    def format_disputed(self, disputed: list) -> str:
        if not disputed:
            return "لا يوجد"
        lines = []
        for item in disputed:
            lines.append(f"- موضوع النزاع: {item.get('subject', '')}")
            for pos in item.get("positions", []):
                bullets_text = "; ".join(pos.get("bullets", []))
                sources_str = ", ".join(pos.get("sources", []))
                lines.append(
                    f"  * {pos.get('party', '')}: {bullets_text} [المصادر: {sources_str}]"
                )
        return "\n".join(lines)

    def format_party_specific(self, party_specific: list) -> str:
        if not party_specific:
            return "لا يوجد"
        lines = []
        for item in party_specific:
            sources_str = ", ".join(item.get("sources", []))
            lines.append(
                f"- [{item.get('party', '')}] {item.get('text', '')} [المصادر: {sources_str}]"
            )
        return "\n".join(lines)

    def collect_sources(self, theme_cluster: dict) -> List[str]:
        """Gather all unique citations from a theme cluster."""
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
    ) -> list:
        """Build system + human messages directly (S2-4: no ChatPromptTemplate)."""
        system_content = _SYSTEM_TEMPLATE.format(theme=theme, role=role)
        human_content = (
            f'الموضوع: "{theme}" ضمن "{role}"\n\n'
            f"النقاط المتفق عليها:\n{agreed_text}\n\n"
            f"النقاط المتنازع عليها:\n{disputed_text}\n\n"
            f"النقاط الخاصة بكل طرف:\n{party_specific_text}\n\n"
            "اكتب ملخصاً في 2-3 فقرات."
        )
        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def build_fallback_summary(self, theme_cluster: dict) -> str:
        """Build a raw-text fallback summary when LLM fails."""
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
        """Extract dispute subjects directly from DisputedPoint data."""
        return [item.get("subject", "") for item in disputed if item.get("subject")]

    def synthesize_theme(self, theme_cluster: dict, role: str) -> dict:
        """Process one theme cluster into a ThemeSummary dict."""
        theme_name = theme_cluster.get("theme_name", "")
        agreed = theme_cluster.get("agreed", [])
        disputed = theme_cluster.get("disputed", [])
        party_specific = theme_cluster.get("party_specific", [])

        # Collect sources before LLM call so they survive fallback paths too
        sources = self.collect_sources(theme_cluster)

        agreed_text = self.format_agreed(agreed)
        disputed_text = self.format_disputed(disputed)
        party_specific_text = self.format_party_specific(party_specific)

        try:
            messages = self._build_messages(
                theme_name, role, agreed_text, disputed_text, party_specific_text
            )
            llm_result = llm_invoke_with_retry(self.parser, messages, logger=logger)

            summary = llm_result.summary
            key_disputes = llm_result.key_disputes

            if not summary or not summary.strip():
                logger.warning("Empty summary for theme '%s', using fallback.", theme_name)
                summary = self.build_fallback_summary(theme_cluster)

            if disputed and not key_disputes:
                logger.warning(
                    "No key disputes for theme '%s', extracting from data.", theme_name
                )
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

    def process_role(self, themed_role: dict) -> dict:
        """Process all themes for one role.

        S1-2: Themes are synthesized concurrently using a thread pool, since
        each call is independent and LangChain clients are thread-safe.
        """
        role = themed_role.get("role", "غير محدد")
        themes = themed_role.get("themes", [])

        logger.info("  Role '%s': %d theme(s) to synthesize", role, len(themes))

        if not themes:
            return {"role": role, "theme_summaries": []}

        # Dispatch all theme synthesis tasks concurrently
        results: List[Any] = [None] * len(themes)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(self.synthesize_theme, theme_cluster, role): i
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
                    logger.error("Unexpected error for theme '%s': %s", theme_name, e)
                    results[idx] = {
                        "theme": theme_name,
                        "summary": self.build_fallback_summary(theme_cluster),
                        "key_disputes": self.extract_dispute_subjects(
                            theme_cluster.get("disputed", [])
                        ),
                        "sources": self.collect_sources(theme_cluster),
                    }

        # Filter None (shouldn't happen) and preserve original order
        theme_summaries = [r for r in results if r is not None]

        return {"role": role, "theme_summaries": theme_summaries}

    def process(self, inputs: dict) -> dict:
        """
        Input:  {"themed_roles": [ThemedRole dicts]}
        Output: {"role_theme_summaries": [RoleThemeSummaries dicts]}
        """
        themed_roles = inputs.get("themed_roles", [])
        if not themed_roles:
            return {"role_theme_summaries": []}

        logger.info("--- Node 4B: Theme-Level Synthesis ---")

        role_theme_summaries = []
        for themed_role in themed_roles:
            role_summary = self.process_role(themed_role)
            role_theme_summaries.append(role_summary)

        return {"role_theme_summaries": role_theme_summaries}
