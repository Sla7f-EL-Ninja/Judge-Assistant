from typing import Literal, List, Optional
from pydantic import BaseModel, Field

# Supported Legal Roles for Metadata (Not classification of content, but document identity)
DocTypeEnum = Literal[
    "صحيفة دعوى",      # Statement of Claim
    "مذكرة دفاع",      # Defense Memo
    "مذكرة رد",        # Reply Memo
    "حافظة مستندات",   # Evidence Portfolio
    "محضر جلسة",       # Hearing Minutes
    "حكم تمهيدي",      # Preliminary Judgment
    "غير محدد"         # Unknown
]

PartyEnum = Literal[
    "المدعي",          # Plaintiff
    "المدعى عليه",     # Defendant
    "النيابة",         # Prosecution
    "المحكمة",         # The Court
    "خبير",            # Expert
    "غير محدد"         # Unknown/Neutral
]

class NormalizedChunk(BaseModel):
    """Atomic unit of text for the pipeline."""
    chunk_id: str = Field(..., description="Unique hash of the content")
    doc_id: str = Field(..., description="Parent document ID")
    
    # Traceability
    page_number: int = Field(..., description="1-based page number")
    paragraph_number: int = Field(..., description="1-based paragraph index on the page")
    
    # Content
    clean_text: str = Field(..., description="Text content with headers/stamps removed")
    
    # Context Metadata (Extracted in Node 0)
    doc_type: DocTypeEnum = Field(..., description="Type of the parent document")
    party: PartyEnum = Field(..., description="The party submitting/authoring this document")

class Node0Output(BaseModel):
    chunks: List[NormalizedChunk]

class DocumentMetadata(BaseModel):
    doc_type: DocTypeEnum = Field(description="The legal type of the document")
    party: PartyEnum = Field(description="The party presenting the document (e.g., Plaintiff, Defendant)")

# The 6 Core Legal Roles + 'Other'
LegalRoleEnum = Literal[
    "الوقائع",          # Facts: The story, events, timeline
    "الطلبات",          # Requests: What they want the court to rule
    "الدفوع",           # Defenses: Legal/procedural arguments against the other side
    "المستندات",        # Evidence: References to attached documents/exhibits
    "الأساس القانوني",  # Legal Basis: Citations of articles/law
    "الإجراءات",        # Procedures: Case history, previous hearings
    "غير محدد"          # Other/Noise: Introductions, pure admin text
]

class ClassifiedChunk(NormalizedChunk):
    """
    Extends NormalizedChunk with semantic classification.
    """
    role: LegalRoleEnum = Field(..., description="The semantic legal category of this text")
    confidence: float = Field(default=1.0, description="Model confidence score (0.0-1.0)")

class Node1Output(BaseModel):
    classified_chunks: List[ClassifiedChunk]


class LegalBullet(BaseModel):
    """Atomic legal idea with full traceability."""
    bullet_id: str = Field(description="Unique ID for this bullet")
    role: LegalRoleEnum = Field(description="Legal role inherited from source chunk")
    bullet: str = Field(description="The atomic legal idea in formal Arabic")
    source: List[str] = Field(description="Citation list, e.g. doc_id ص12 ف3")
    party: PartyEnum = Field(description="Party inherited from source chunk")
    chunk_id: str = Field(description="Back-reference to the source ClassifiedChunk")


class Node2Output(BaseModel):
    bullets: List[LegalBullet]


# --- Node 3 Output Schemas ---

class AgreedBullet(BaseModel):
    """An agreed-upon or uncontested fact in formal Arabic."""
    text: str = Field(description="The agreed-upon fact in formal Arabic")
    sources: List[str] = Field(description="Merged citations from all supporting bullets")


class DisputePosition(BaseModel):
    """One party's position within a disputed point."""
    party: PartyEnum = Field(description="The party taking this position")
    bullets: List[str] = Field(description="Exact original bullet texts from this party")
    sources: List[str] = Field(description="Citations for these bullets")


class DisputedPoint(BaseModel):
    """A point of contention between parties."""
    subject: str = Field(description="Brief label for what is disputed")
    positions: List[DisputePosition] = Field(description="Each party's position with exact text")


class PartyBullet(BaseModel):
    """A point unique to one party, not contested or matched."""
    party: PartyEnum = Field(description="The party this point belongs to")
    text: str = Field(description="The bullet text")
    sources: List[str] = Field(description="Citations")


class RoleAggregation(BaseModel):
    """Complete aggregation output for one legal role."""
    role: LegalRoleEnum = Field(description="The legal role this aggregation covers")
    agreed: List[AgreedBullet]
    disputed: List[DisputedPoint]
    party_specific: List[PartyBullet]


class Node3Output(BaseModel):
    """Full output of Node 3: per-role aggregations."""
    role_aggregations: List[RoleAggregation]


# --- Node 4A Output Schemas ---

class ThemeCluster(BaseModel):
    """One thematic group within a legal role."""
    theme_name: str = Field(description="Descriptive theme name in Arabic")
    agreed: List[AgreedBullet] = Field(default_factory=list)
    disputed: List[DisputedPoint] = Field(default_factory=list)
    party_specific: List[PartyBullet] = Field(default_factory=list)
    bullet_count: int = Field(description="Total items in this theme cluster")


class ThemedRole(BaseModel):
    """A legal role with its items organized into thematic clusters."""
    role: LegalRoleEnum = Field(description="The legal role")
    themes: List[ThemeCluster] = Field(description="3-7 thematic clusters")


class Node4AOutput(BaseModel):
    """Full output of Node 4A."""
    themed_roles: List[ThemedRole]


# --- Node 4B Output Schemas ---

class ThemeSummary(BaseModel):
    """Synthesized summary for one theme within a role."""
    theme: str = Field(description="Theme name from clustering")
    summary: str = Field(description="2-3 paragraphs in formal legal Arabic")
    key_disputes: List[str] = Field(
        default_factory=list,
        description="Brief labels for main disputes in this theme"
    )
    sources: List[str] = Field(description="All citations referenced in the summary")


class RoleThemeSummaries(BaseModel):
    """All theme summaries for one legal role."""
    role: LegalRoleEnum
    theme_summaries: List[ThemeSummary]


class Node4BOutput(BaseModel):
    """Full output of Node 4B."""
    role_theme_summaries: List[RoleThemeSummaries]


# --- Node 5 Output Schemas ---

class CaseBrief(BaseModel):
    """The 7 sections of the judge-facing brief."""
    dispute_summary: str = Field(
        description="ملخص النزاع: فقرة واحدة تلخص جوهر النزاع بين الأطراف"
    )
    uncontested_facts: str = Field(
        description="الوقائع غير المتنازع عليها: الوقائع التي لم ينازع فيها أي طرف"
    )
    key_disputes: str = Field(
        description="نقاط الخلاف الجوهرية: النقاط التي يتناقض فيها الأطراف"
    )
    party_requests: str = Field(
        description="طلبات الخصوم: ما يطلبه كل طرف من المحكمة"
    )
    party_defenses: str = Field(
        description="دفوع الخصوم: الدفوع الشكلية والموضوعية لكل طرف"
    )
    submitted_documents: str = Field(
        description="المستندات المقدمة: المستندات التي قدمها كل طرف"
    )
    legal_questions: str = Field(
        description="الأسئلة القانونية المطروحة: المسائل القانونية التي تحتاج فصلاً"
    )


class Node5Output(BaseModel):
    """Full output of Node 5."""
    case_brief: CaseBrief
    all_sources: List[str] = Field(
        description="All unique citations referenced across the brief"
    )
    rendered_brief: str = Field(
        description="The full brief rendered as Arabic markdown text"
    )
