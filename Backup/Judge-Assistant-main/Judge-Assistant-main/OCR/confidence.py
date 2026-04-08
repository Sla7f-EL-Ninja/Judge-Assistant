"""
OCR.confidence
--------------
Confidence scoring from token log-probabilities.

Provides page-level confidence (primary) and per-word confidence (stretch goal).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from OCR.models import WordConfidence

logger = logging.getLogger(__name__)


def compute_page_confidence(
    scores: Tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
) -> float:
    """Compute page-level confidence from generation scores.

    For each generated token, applies softmax to the logits and takes the
    probability of the actually-generated token ID.  Returns the average
    across all tokens (0.0 to 1.0).

    Parameters
    ----------
    scores:
        Tuple of logit tensors, one per generated token.  Each tensor has
        shape ``(batch_size, vocab_size)``.
    generated_ids:
        The token IDs that were actually generated, shape ``(num_tokens,)``.

    Returns
    -------
    float
        Average token probability (page-level confidence).
    """
    if not scores or len(generated_ids) == 0:
        return 0.0

    token_probs: List[float] = []

    for step_idx, logits in enumerate(scores):
        if step_idx >= len(generated_ids):
            break

        # logits shape: (batch_size, vocab_size) -- take first batch element
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
    tokenizer: PreTrainedTokenizerBase,
) -> Optional[List[WordConfidence]]:
    """Compute per-word confidence by grouping tokens into words.

    This is a best-effort stretch goal.  Arabic tokenization is complex,
    so word boundaries are detected by looking for whitespace in decoded
    token text.

    Parameters
    ----------
    scores:
        Tuple of logit tensors from generation.
    generated_ids:
        The generated token IDs.
    tokenizer:
        The tokenizer used for decoding tokens back to text.

    Returns
    -------
    list[WordConfidence] or None
        Per-word confidence scores, or ``None`` if computation fails.
    """
    if not scores or len(generated_ids) == 0:
        return None

    try:
        # Collect (decoded_text, probability) for each token
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

        # Group consecutive tokens into words by whitespace boundaries
        words: List[WordConfidence] = []
        current_word_tokens: List[str] = []
        current_word_probs: List[float] = []

        for text, prob in token_entries:
            if not text.strip():
                # Whitespace token -- flush current word
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

            # Check if this token starts with whitespace (word boundary)
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

        # Flush remaining tokens
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
