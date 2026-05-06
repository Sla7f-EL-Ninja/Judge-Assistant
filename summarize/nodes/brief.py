import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from summarize.schemas import CaseBrief, Node5Output
from summarize.utils import get_logger, llm_invoke_with_retry
from summarize.prompts.brief import SYSTEM_PROMPT, BIAS_KEYWORDS

logger = get_logger("hakim.node_5")


class Node5_BriefGenerator:

    def __init__(self, llm):
        self.llm = llm
        self.parser = llm.with_structured_output(CaseBrief)

    def organize_by_role(self, input_data: dict) -> Dict[str, List[dict]]:
        role_map: Dict[str, List[dict]] = {}
        for rts in input_data.get("role_theme_summaries", []):
            role = rts.get("role", "غير محدد")
            role_map[role] = rts.get("theme_summaries", [])
        return role_map

    def compile_key_disputes(self, input_data: dict) -> List[str]:
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
        parts: List[str] = []
        for role, themes in role_map.items():
            if role == "غير محدد":
                continue
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

    def _build_messages(
        self,
        role_summaries_text: str,
        compiled_key_disputes: str,
        party_manifest: Optional[Dict] = None,
    ) -> list:
        absent_clause = ""
        if party_manifest:
            parties_without_defense = [
                p for p, doc_types in party_manifest.items()
                if "مذكرة دفاع" not in doc_types
                and "مذكرة رد" not in doc_types
                and p not in ("المحكمة", "غير محدد", "النيابة", "خبير")
            ]
            if parties_without_defense:
                names = "، ".join(parties_without_defense)
                absent_clause = (
                    f"\n\nتنبيه حاسم — أطراف بلا مذكرات دفاع:\n"
                    f"الأطراف التالية لم يقدموا مذكرات دفاع وفقاً لوثائق الملف: {names}.\n"
                    f"في قسم 'دفوع الخصوم' (القسم 5)، اكتب لكل طرف من هؤلاء حرفياً: "
                    f"'لم ترد مذكرة دفاع لهذا الطرف في الملف المقدم' — "
                    f"يُمنع منعاً باتاً اختلاق أي دفع أو موقف لهم."
                )

        human_content = (
            "فيما يلي ملخصات الأدوار القانونية للقضية:\n\n"
            f"{role_summaries_text}\n\n"
            "نقاط الخلاف المستخلصة من جميع الأدوار:\n"
            f"{compiled_key_disputes}"
            f"{absent_clause}\n\n"
            "اكتب مذكرة ملخص القضية بالأقسام السبعة المطلوبة."
        )
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    def generate_brief(
        self,
        role_summaries_text: str,
        compiled_key_disputes: str,
        party_manifest: Optional[Dict] = None,
    ) -> CaseBrief:
        messages = self._build_messages(
            role_summaries_text, compiled_key_disputes, party_manifest
        )
        return llm_invoke_with_retry(self.parser, messages, logger=logger)

    def validate_brief(self, brief: CaseBrief) -> bool:
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

    def build_fallback_brief(
        self,
        role_map: Dict[str, List[dict]],
        key_disputes: List[str],
    ) -> CaseBrief:
        fallback_prefix = "[ملخص خام - يحتاج مراجعة]\n"
        no_info = "لا تتوفر معلومات كافية"

        summary_parts = []
        for role, themes in role_map.items():
            if role == "غير محدد":
                continue
            if themes:
                first_summary = themes[0].get("summary", "")
                first_sentence = first_summary.split(".")[0] if first_summary else ""
                if first_sentence:
                    summary_parts.append(f"{role}: {first_sentence}.")
        dispute_summary = (
            fallback_prefix + "\n".join(summary_parts) if summary_parts else no_info
        )

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

        key_disputes_text = (
            fallback_prefix + "\n".join(f"- {d}" for d in key_disputes)
            if key_disputes
            else no_info
        )

        talabat_themes = role_map.get("الطلبات", [])
        if talabat_themes:
            req_parts = [ts.get("summary", "") for ts in talabat_themes if ts.get("summary")]
            party_requests = (
                fallback_prefix + "\n\n".join(req_parts) if req_parts else no_info
            )
        else:
            party_requests = "لا تتوفر معلومات كافية عن طلبات الخصوم"

        dofoo_themes = role_map.get("الدفوع", [])
        if dofoo_themes:
            def_parts = [ts.get("summary", "") for ts in dofoo_themes if ts.get("summary")]
            party_defenses = (
                fallback_prefix + "\n\n".join(def_parts) if def_parts else no_info
            )
        else:
            party_defenses = "لا تتوفر معلومات كافية عن دفوع الخصوم"

        mostanad_themes = role_map.get("المستندات", [])
        if mostanad_themes:
            doc_parts = [ts.get("summary", "") for ts in mostanad_themes if ts.get("summary")]
            submitted_documents = (
                fallback_prefix + "\n\n".join(doc_parts) if doc_parts else no_info
            )
        else:
            submitted_documents = "لا تتوفر معلومات كافية عن المستندات المقدمة"

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

    def _enforce_absent_party_defenses(
        self, brief: CaseBrief, party_manifest: Optional[Dict]
    ) -> CaseBrief:
        if not party_manifest:
            return brief

        parties_without_defense = [
            p for p, doc_types in party_manifest.items()
            if "مذكرة دفاع" not in doc_types
            and "مذكرة رد" not in doc_types
            and p not in ("المحكمة", "غير محدد", "النيابة", "خبير")
        ]

        if not parties_without_defense:
            return brief

        disclaimer = "لم ترد مذكرة دفاع لهذا الطرف في الملف المقدم"
        defenses_text = brief.party_defenses
        corrections = []

        for party in parties_without_defense:
            if party not in defenses_text:
                continue
            pattern = re.compile(
                re.escape(party) + r'.{50,}',
                re.DOTALL,
            )
            if pattern.search(defenses_text) and disclaimer not in defenses_text:
                logger.warning(
                    "Fix 3: '%s' appears in party_defenses without disclaimer — injecting correction.",
                    party,
                )
                corrections.append(f"\n[تصحيح] {party}: {disclaimer}")

        if corrections:
            brief.party_defenses = defenses_text + "\n" + "\n".join(corrections)

        return brief

    def render_brief(
        self,
        brief: CaseBrief,
        all_sources: List[str],
        ghayr_summaries: Optional[List[str]] = None,
    ) -> str:
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

        if ghayr_summaries:
            lines.append("## ثامناً: معلومات إضافية (غير مصنفة)")
            lines.append("\n\n".join(ghayr_summaries))
            lines.append("")

        lines.append("---")
        lines.append(
            f"المصادر المرجعية: {', '.join(all_sources)}"
            if all_sources
            else "المصادر المرجعية: لا توجد مصادر"
        )

        return "\n".join(lines)
    
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
        party_manifest: Dict = inputs.get("party_manifest", {})

        logger.info("  Roles present: %s", list(role_map.keys()))
        logger.info("  Key disputes compiled: %d", len(key_disputes))
        logger.info("  Total unique sources: %d", len(all_sources))
        if party_manifest:
            logger.info("  Party manifest: %s", list(party_manifest.keys()))

        if not all_sources:
            logger.warning("No source citations found in input.")

        # Extract غير محدد BEFORE building the LLM prompt — it's rendered separately
        ghayr_themes = role_map.get("غير محدد", [])
        ghayr_summaries = [ts.get("summary", "") for ts in ghayr_themes if ts.get("summary")]
        if ghayr_summaries:
            logger.info("  غير محدد: %d theme(s) extracted for standalone section.", len(ghayr_summaries))

        role_summaries_text, compiled_key_disputes = self.build_context_for_prompt(
            role_map, key_disputes
        )

        try:
            brief = self.generate_brief(
                role_summaries_text, compiled_key_disputes, party_manifest
            )
            brief = self._enforce_absent_party_defenses(brief, party_manifest)

            if not self.validate_brief(brief):
                logger.warning("Validation failed, using fallback assembly.")
                brief = self.build_fallback_brief(role_map, key_disputes)
            else:
                logger.info("Brief generated and validated successfully.")

        except Exception as e:
            logger.error("LLM call failed: %s — using fallback assembly.", e)
            brief = self.build_fallback_brief(role_map, key_disputes)

        rendered = self.render_brief(brief, all_sources, ghayr_summaries or None)

        return {
            "case_brief": brief.model_dump(),
            "all_sources": all_sources,
            "rendered_brief": rendered,
        }
