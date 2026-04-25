"""Case Reasoner configuration — reads from settings.yaml case_reasoner section."""
from config import cfg

_section: dict = cfg.get("case_reasoner") or {}
_conf: dict = _section.get("confidence") or {}
_weights: dict = _conf.get("weights") or {}
_thresholds: dict = _conf.get("thresholds") or {}

CONFIDENCE_WEIGHTS: dict = {
    "unsupported_ratio":        float(_weights.get("unsupported_ratio", 0.25)),
    "disputed_ratio":           float(_weights.get("disputed_ratio", 0.15)),
    "insufficient_ratio":       float(_weights.get("insufficient_ratio", 0.20)),
    "citation_failure_ratio":   float(_weights.get("citation_failure_ratio", 0.15)),
    "logical_issues":           float(_weights.get("logical_issues", 0.10)),
    "completeness_gap":         float(_weights.get("completeness_gap", 0.10)),
    "reconciliation_triggered": float(_weights.get("reconciliation_triggered", 0.05)),
}

CONFIDENCE_THRESHOLDS: dict = {
    "high":   float(_thresholds.get("high", 0.75)),
    "medium": float(_thresholds.get("medium", 0.45)),
}
