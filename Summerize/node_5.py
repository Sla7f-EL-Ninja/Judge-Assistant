import sys
import os
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schemas import CaseBrief, Node5Output
from utils import get_logger, llm_invoke_with_retry

logger = get_logger("hakim.node_5")


# --- Static prompt content (no user content embedded) ---

_SYSTEM_PROMPT = """أنت مساعد قضائي متخصص في إعداد ملخصات القضايا للقاضي قبل الجلسة.

مهمتك: كتابة مذكرة ملخص قضية من 7 أقسام بناءً على ملخصات الأدوار القانونية المقدمة.

الأقسام المطلوبة:
1. ملخص النزاع: فقرة واحدة موجزة تصف جوهر النزاع وأطرافه وموضوعه
2. الوقائع غير المتنازع عليها: الوقائع الثابتة التي لم ينازع فيها أي طرف
3. نقاط الخلاف الجوهرية: كل نقطة خلاف مع بيان موقف كل طرف باختصار
4. طلبات الخصوم: ما يطلبه كل طرف من المحكمة، مفصولاً بحسب الطرف
5. دفوع الخصوم: الدفوع الشكلية والموضوعية لكل طرف
6. المستندات المقدمة: قائمة المستندات مع نسبتها للطرف المقدم
7. الأسئلة القانونية المطروحة: المسائل القانونية التي يثيرها النزاع وتحتاج فصلاً من المحكمة

شروط صارمة:
- استخدم اللغة العربية القانونية الرسمية فقط
- لا تضف أي رأي أو استنتاج أو توصية أو اتجاه للحكم
- لا تختلق وقائع أو نصوص قانونية غير موجودة في المدخلات
- حافظ على الحياد التام بين الأطراف
- إذا لم تتوفر معلومات لقسم معين، اكتب "لا تتوفر معلومات كافية"
- استخدم صيغ مثل: "يتمسك... بينما يدفع..." عند عرض الخلافات"""

# Bias keywords to check for in validation
BIAS_KEYWORDS = ["نوصي", "يجب على المحكمة", "نرى أن", "نقترح", "ينبغي الحكم"]


class Node5_BriefGenerator:

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(CaseBrief)

    # --- Data organisation ---

    def organize_by_role(self, input_data: dict) -> Dict[str, List[dict]]:
        """Returns {role_name: [theme_summaries]} from Node4BOutput."""
        role_map: Dict[str, List[dict]] = {}
        for rts in input_data.get("role_theme_summaries", []):
            role = rts.get("role", "غير محدد")
            role_map[role] = rts.get("theme_summaries", [])
        return role_map

    def compile_key_disputes(self, input_data: dict) -> List[str]:
        """Collect all key_disputes strings across all roles, deduplicated."""
        disputes: List[str] = []
        seen: set = set()
        for rts in input_data.get("role_theme_summaries", []):
            for ts in rts.get("theme_summaries", []):
                for d in ts.get("key_disputes", []):
                    if d and d not in seen:
                        seen.add(d)
                        disputes.append(d)
        return disputes

    def collect_all_sources(self, input_data: dict) -> List[str]:
        """Collect all unique source citations across the entire input."""
        sources: List[str] = []
        seen: set = set()
        for rts in input_data.get("role_theme_summaries", []):
            for ts in rts.get("theme_summaries", []):
                for s in ts.get("sources", []):
                    if s and s not in seen:
                        seen.add(s)
                        sources.append(s)
        return sources

    def build_context_for_prompt(
        self,
        role_map: Dict[str, List[dict]],
        key_disputes: List[str],
    ) -> tuple:
        """Format all role summaries + disputes into text blocks for the LLM.

        Returns (role_summaries_text, compiled_key_disputes_text).
        """
        parts: List[str] = []
        for role, themes in role_map.items():
            parts.append(f"=== {role} ===")
            parts.append("")
            for ts in themes:
                theme = ts.get("theme", "")
                summary = ts.get("summary", "")
                sources = ts.get("sources", [])
                parts.append(f"-- {theme} --")
                parts.append(summary)
                if sources:
                    parts.append(f"المصادر: {', '.join(sources)}")
                parts.append("")

        role_summaries_text = "\n".join(parts)

        compiled_key_disputes = (
            "\n".join(f"- {d}" for d in key_disputes)
            if key_disputes
            else "لا توجد نقاط خلاف مستخلصة"
        )

        return role_summaries_text, compiled_key_disputes

    # --- LLM call ---

    def _build_messages(
        self, role_summaries_text: str, compiled_key_disputes: str
    ) -> list:
        """Build system + human messages directly (S2-4: no ChatPromptTemplate)."""
        human_content = (
            "فيما يلي ملخصات الأدوار القانونية للقضية:\n\n"
            f"{role_summaries_text}\n\n"
            "نقاط الخلاف المستخلصة من جميع الأدوار:\n"
            f"{compiled_key_disputes}\n\n"
            "اكتب مذكرة ملخص القضية بالأقسام السبعة المطلوبة."
        )
        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    def generate_brief(
        self, role_summaries_text: str, compiled_key_disputes: str
    ) -> CaseBrief:
        """Single LLM call with structured output to generate the 7-section brief."""
        messages = self._build_messages(role_summaries_text, compiled_key_disputes)
        return llm_invoke_with_retry(self.parser, messages, logger=logger)

    # --- Validation ---

    def validate_brief(self, brief: CaseBrief) -> bool:
        """Validate all 7 fields are non-empty and contain no bias language."""
        fields = [
            brief.dispute_summary,
            brief.uncontested_facts,
            brief.key_disputes,
            brief.party_requests,
            brief.party_defenses,
            brief.submitted_documents,
            brief.legal_questions,
        ]

        for field in fields:
            if not field or not field.strip():
                return False

        full_text = " ".join(fields)
        for keyword in BIAS_KEYWORDS:
            if keyword in full_text:
                logger.warning("Bias keyword detected: '%s'", keyword)
                return False

        return True

    # --- Fallback ---

    def build_fallback_brief(
        self,
        role_map: Dict[str, List[dict]],
        key_disputes: List[str],
    ) -> CaseBrief:
        """Raw assembly when LLM fails. Each section built from available role data."""
        fallback_prefix = "[ملخص خام - يحتاج مراجعة]\n"
        no_info = "لا تتوفر معلومات كافية"

        # dispute_summary: first sentence of each role's first theme
        summary_parts = []
        for role, themes in role_map.items():
            if themes:
                first_summary = themes[0].get("summary", "")
                first_sentence = first_summary.split(".")[0] if first_summary else ""
                if first_sentence:
                    summary_parts.append(f"{role}: {first_sentence}.")
        dispute_summary = (
            fallback_prefix + "\n".join(summary_parts) if summary_parts else no_info
        )

        # S2-11: uncontested_facts — only include themes from الوقائع that have
        # no key_disputes (i.e. genuinely uncontested themes).
        waqa_themes = role_map.get("الوقائع", [])
        if waqa_themes:
            uncontested_parts = [
                ts.get("summary", "")
                for ts in waqa_themes
                if ts.get("summary") and not ts.get("key_disputes")
            ]
            uncontested_facts = (
                fallback_prefix + "\n\n".join(uncontested_parts)
                if uncontested_parts
                else no_info
            )
        else:
            uncontested_facts = no_info

        # key_disputes
        key_disputes_text = (
            fallback_prefix + "\n".join(f"- {d}" for d in key_disputes)
            if key_disputes
            else no_info
        )

        # party_requests from الطلبات
        talabat_themes = role_map.get("الطلبات", [])
        if talabat_themes:
            req_parts = [ts.get("summary", "") for ts in talabat_themes if ts.get("summary")]
            party_requests = (
                fallback_prefix + "\n\n".join(req_parts) if req_parts else no_info
            )
        else:
            party_requests = "لا تتوفر معلومات كافية عن طلبات الخصوم"

        # party_defenses from الدفوع
        dofoo_themes = role_map.get("الدفوع", [])
        if dofoo_themes:
            def_parts = [ts.get("summary", "") for ts in dofoo_themes if ts.get("summary")]
            party_defenses = (
                fallback_prefix + "\n\n".join(def_parts) if def_parts else no_info
            )
        else:
            party_defenses = "لا تتوفر معلومات كافية عن دفوع الخصوم"

        # submitted_documents from المستندات
        mostanad_themes = role_map.get("المستندات", [])
        if mostanad_themes:
            doc_parts = [ts.get("summary", "") for ts in mostanad_themes if ts.get("summary")]
            submitted_documents = (
                fallback_prefix + "\n\n".join(doc_parts) if doc_parts else no_info
            )
        else:
            submitted_documents = "لا تتوفر معلومات كافية عن المستندات المقدمة"

        # legal_questions from الأساس القانوني
        legal_themes = role_map.get("الأساس القانوني", [])
        legal_disputes: List[str] = []
        for ts in legal_themes:
            legal_disputes.extend(ts.get("key_disputes", []))
        if legal_disputes:
            legal_questions = fallback_prefix + "\n".join(f"- {d}" for d in legal_disputes)
        elif key_disputes:
            legal_questions = fallback_prefix + "\n".join(f"- {d}" for d in key_disputes)
        else:
            legal_questions = no_info

        return CaseBrief(
            dispute_summary=dispute_summary,
            uncontested_facts=uncontested_facts,
            key_disputes=key_disputes_text,
            party_requests=party_requests,
            party_defenses=party_defenses,
            submitted_documents=submitted_documents,
            legal_questions=legal_questions,
        )

    # --- Rendering ---

    def render_brief(self, brief: CaseBrief, all_sources: List[str]) -> str:
        """Convert CaseBrief to Arabic markdown string."""
        sections = [
            ("أولاً: ملخص النزاع", brief.dispute_summary),
            ("ثانياً: الوقائع غير المتنازع عليها", brief.uncontested_facts),
            ("ثالثاً: نقاط الخلاف الجوهرية", brief.key_disputes),
            ("رابعاً: طلبات الخصوم", brief.party_requests),
            ("خامساً: دفوع الخصوم", brief.party_defenses),
            ("سادساً: المستندات المقدمة", brief.submitted_documents),
            ("سابعاً: الأسئلة القانونية المطروحة", brief.legal_questions),
        ]

        lines = ["# مذكرة ملخص القضية", ""]
        for title, content in sections:
            lines.append(f"## {title}")
            lines.append(content)
            lines.append("")

        lines.append("---")
        lines.append(
            f"المصادر المرجعية: {', '.join(all_sources)}"
            if all_sources
            else "المصادر المرجعية: لا توجد مصادر"
        )

        return "\n".join(lines)

    # --- Entry point ---

    def process(self, inputs: dict) -> dict:
        """
        Input:  {"role_theme_summaries": [RoleThemeSummaries dicts]}
        Output: {"case_brief": dict, "all_sources": [...], "rendered_brief": str}
        """
        role_theme_summaries = inputs.get("role_theme_summaries", [])
        if not role_theme_summaries:
            logger.warning("Empty input to Node 5, producing empty brief.")
            empty_brief = CaseBrief(
                dispute_summary="لا تتوفر معلومات كافية",
                uncontested_facts="لا تتوفر معلومات كافية",
                key_disputes="لا تتوفر معلومات كافية",
                party_requests="لا تتوفر معلومات كافية",
                party_defenses="لا تتوفر معلومات كافية",
                submitted_documents="لا تتوفر معلومات كافية",
                legal_questions="لا تتوفر معلومات كافية",
            )
            return {
                "case_brief": empty_brief.model_dump(),
                "all_sources": [],
                "rendered_brief": self.render_brief(empty_brief, []),
            }

        logger.info("--- Node 5: Judge-Facing Case Brief ---")

        role_map = self.organize_by_role(inputs)
        key_disputes = self.compile_key_disputes(inputs)
        all_sources = self.collect_all_sources(inputs)

        logger.info("  Roles present: %s", list(role_map.keys()))
        logger.info("  Key disputes compiled: %d", len(key_disputes))
        logger.info("  Total unique sources: %d", len(all_sources))

        if not all_sources:
            logger.warning("No source citations found in input.")

        role_summaries_text, compiled_key_disputes = self.build_context_for_prompt(
            role_map, key_disputes
        )

        try:
            brief = self.generate_brief(role_summaries_text, compiled_key_disputes)

            if not self.validate_brief(brief):
                logger.warning("Validation failed, using fallback assembly.")
                brief = self.build_fallback_brief(role_map, key_disputes)
            else:
                logger.info("Brief generated and validated successfully.")

        except Exception as e:
            logger.error("LLM call failed: %s — using fallback assembly.", e)
            brief = self.build_fallback_brief(role_map, key_disputes)

        rendered = self.render_brief(brief, all_sources)

        return {
            "case_brief": brief.model_dump(),
            "all_sources": all_sources,
            "rendered_brief": rendered,
        }
