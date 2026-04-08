import sys
import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Add parent directory to path for shared schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import (
    LegalRoleEnum,
    ThemeSummary, RoleThemeSummaries, Node4BOutput,
)


# --- Internal LLM Schema ---

class SynthesisResultLLM(BaseModel):
    """LLM output: synthesis for one theme."""
    summary: str = Field(description="ملخص الموضوع في 2-3 فقرات")
    key_disputes: List[str] = Field(description="عناوين مختصرة لنقاط الخلاف الجوهرية")


# --- Prompt Templates ---

SYSTEM_PROMPT_4B = """أنت مساعد قضائي متخصص في تلخيص المعلومات القانونية للقاضي.

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

HUMAN_TEMPLATE_4B = """الموضوع: "{theme}" ضمن "{role}"

النقاط المتفق عليها:
{agreed_text}

النقاط المتنازع عليها:
{disputed_text}

النقاط الخاصة بكل طرف:
{party_specific_text}

اكتب ملخصاً في 2-3 فقرات."""


class Node4B_ThemeSynthesis:

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(SynthesisResultLLM)

    def format_agreed(self, agreed: list) -> str:
        """Format agreed items for prompt. Returns 'لا يوجد' if empty."""
        if not agreed:
            return "لا يوجد"
        lines = []
        for item in agreed:
            sources_str = ", ".join(item.get("sources", []))
            lines.append(f"- {item.get('text', '')} [المصادر: {sources_str}]")
        return "\n".join(lines)

    def format_disputed(self, disputed: list) -> str:
        """Format disputed items showing both sides. Returns 'لا يوجد' if empty."""
        if not disputed:
            return "لا يوجد"
        lines = []
        for item in disputed:
            lines.append(f"- موضوع النزاع: {item.get('subject', '')}")
            for pos in item.get("positions", []):
                party = pos.get("party", "")
                bullets_text = "; ".join(pos.get("bullets", []))
                sources_str = ", ".join(pos.get("sources", []))
                lines.append(f"  * {party}: {bullets_text} [المصادر: {sources_str}]")
        return "\n".join(lines)

    def format_party_specific(self, party_specific: list) -> str:
        """Format party-specific items with party labels. Returns 'لا يوجد' if empty."""
        if not party_specific:
            return "لا يوجد"
        lines = []
        for item in party_specific:
            party = item.get("party", "")
            sources_str = ", ".join(item.get("sources", []))
            lines.append(f"- [{party}] {item.get('text', '')} [المصادر: {sources_str}]")
        return "\n".join(lines)

    def collect_sources(self, theme_cluster: dict) -> List[str]:
        """Gather all unique citations from a theme cluster."""
        sources = []
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

    def create_prompt_messages(
        self, theme: str, role: str,
        agreed_text: str, disputed_text: str,
        party_specific_text: str,
    ) -> list:
        """Build system + human messages."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_4B),
            ("human", HUMAN_TEMPLATE_4B),
        ])
        return prompt.format_messages(
            theme=theme,
            role=role,
            agreed_text=agreed_text,
            disputed_text=disputed_text,
            party_specific_text=party_specific_text,
        )

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
        """Process one theme cluster into a ThemeSummary."""
        theme_name = theme_cluster.get("theme_name", "")
        agreed = theme_cluster.get("agreed", [])
        disputed = theme_cluster.get("disputed", [])
        party_specific = theme_cluster.get("party_specific", [])

        # Collect sources programmatically (not from LLM)
        sources = self.collect_sources(theme_cluster)

        # Format sections for the prompt
        agreed_text = self.format_agreed(agreed)
        disputed_text = self.format_disputed(disputed)
        party_specific_text = self.format_party_specific(party_specific)

        try:
            messages = self.create_prompt_messages(
                theme_name, role,
                agreed_text, disputed_text, party_specific_text,
            )
            llm_result = self.parser.invoke(messages)

            summary = llm_result.summary
            key_disputes = llm_result.key_disputes

            # Validation: non-empty summary
            if not summary or not summary.strip():
                print(f"  Warning: empty summary for theme '{theme_name}', using fallback.")
                summary = self.build_fallback_summary(theme_cluster)

            # Validation: key disputes present when disputed items exist
            if disputed and not key_disputes:
                print(f"  Warning: no key disputes returned for theme '{theme_name}', extracting from data.")
                key_disputes = self.extract_dispute_subjects(disputed)

            return {
                "theme": theme_name,
                "summary": summary,
                "key_disputes": key_disputes,
                "sources": sources,
            }

        except Exception as e:
            print(f"  Error in LLM call for theme '{theme_name}': {e}")
            # Fallback: raw text summary
            key_disputes = self.extract_dispute_subjects(disputed)
            return {
                "theme": theme_name,
                "summary": self.build_fallback_summary(theme_cluster),
                "key_disputes": key_disputes,
                "sources": sources,
            }

    def process_role(self, themed_role: dict) -> dict:
        """Process all themes for one role."""
        role = themed_role.get("role", "غير محدد")
        themes = themed_role.get("themes", [])

        print(f"  Role '{role}': {len(themes)} theme(s) to synthesize")

        theme_summaries = []
        for theme_cluster in themes:
            theme_name = theme_cluster.get("theme_name", "")
            print(f"    Synthesizing theme: '{theme_name}' "
                  f"({theme_cluster.get('bullet_count', 0)} items)...")
            summary = self.synthesize_theme(theme_cluster, role)
            theme_summaries.append(summary)

        return {
            "role": role,
            "theme_summaries": theme_summaries,
        }

    def process(self, inputs: dict) -> dict:
        """Entry point.
        Input: Node4AOutput dict with {"themed_roles": [...]}
        Output: Node4BOutput dict with {"role_theme_summaries": [...]}
        """
        themed_roles = inputs.get("themed_roles", [])
        if not themed_roles:
            return {"role_theme_summaries": []}

        print("\n--- Node 4B: Theme-Level Synthesis ---")
        role_theme_summaries = []
        for themed_role in themed_roles:
            role_summary = self.process_role(themed_role)
            role_theme_summaries.append(role_summary)

        return {"role_theme_summaries": role_theme_summaries}
