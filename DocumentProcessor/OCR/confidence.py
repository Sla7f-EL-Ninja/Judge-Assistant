"""
DocumentProcessor.OCR.confidence
----------------------------------
Confidence scoring from token log-probabilities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from DocumentProcessor.OCR.models import WordConfidence

logger = logging.getLogger(__name__)


def compute_page_confidence(
    scores: Tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
) -> float:
    """Compute page-level confidence from generation scores."""
    if not scores or len(generated_ids) == 0:
        return 0.0

    token_probs: List[float] = []

    for step_idx, logits in enumerate(scores):
        if step_idx >= len(generated_ids):
            break

        probs = torch.softmax(logits[0], dim=-1)
        token_id = generated_ids[step_idx].item()
        token_prob = probs[token_id].item()
        token_probs.append(token_prob)

    if not token_probs:
        return 0.0

    return sum(token_probs) / len(token_probs)


def compute_word_confidences(
    scores: Tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
    tokenizer: "PreTrainedTokenizerBase",
) -> Optional[List[WordConfidence]]:
    """Compute per-word confidence by grouping tokens into words."""
    if not scores or len(generated_ids) == 0:
        return None

    try:
        token_entries: List[Tuple[str, float]] = []

        for step_idx, logits in enumerate(scores):
            if step_idx >= len(generated_ids):
                break

            probs = torch.softmax(logits[0], dim=-1)
            token_id = generated_ids[step_idx].item()
            token_prob = probs[token_id].item()
            token_text = tokenizer.decode(
                [token_id], skip_special_tokens=True,
            )
            token_entries.append((token_text, token_prob))

        if not token_entries:
            return None

        words: List[WordConfidence] = []
        current_word_tokens: List[str] = []
        current_word_probs: List[float] = []

        for text, prob in token_entries:
            if not text.strip():
                if current_word_tokens:
                    word_text = "".join(current_word_tokens).strip()
                    if word_text:
                        avg_prob = sum(current_word_probs) / len(current_word_probs)
                        words.append(WordConfidence(
                            word=word_text,
                            confidence=round(avg_prob, 4),
                        ))
                    current_word_tokens = []
                    current_word_probs = []
                continue

            if text and text[0] in (" ", "\t", "\n"):
                if current_word_tokens:
                    word_text = "".join(current_word_tokens).strip()
                    if word_text:
                        avg_prob = sum(current_word_probs) / len(current_word_probs)
                        words.append(WordConfidence(
                            word=word_text,
                            confidence=round(avg_prob, 4),
                        ))
                    current_word_tokens = []
                    current_word_probs = []

            current_word_tokens.append(text)
            current_word_probs.append(prob)

        if current_word_tokens:
            word_text = "".join(current_word_tokens).strip()
            if word_text:
                avg_prob = sum(current_word_probs) / len(current_word_probs)
                words.append(WordConfidence(
                    word=word_text,
                    confidence=round(avg_prob, 4),
                ))

        return words if words else None

    except Exception as exc:
        logger.warning("Per-word confidence computation failed: %s", exc)
        return None
