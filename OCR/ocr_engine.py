"""
OCR.ocr_engine
--------------
QARI OCR engine: model loading and per-page inference.

Uses a singleton pattern so the model is loaded once and reused.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import torch
from PIL import Image

from OCR.confidence import compute_page_confidence, compute_word_confidences
from OCR.models import WordConfidence

logger = logging.getLogger(__name__)

# Module-level singleton
_engine_instance: Optional[QARIEngine] = None
_engine_lock = threading.Lock()


class QARIEngine:
    """Wrapper around the QARI (Qwen2VL-based) OCR model.

    Loads the model with 8-bit quantization on first instantiation and
    provides a ``ocr_page`` method for per-page inference with confidence
    scoring.
    """

    def __init__(
        self,
        model_name: str = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
        max_new_tokens: int = 4000,
        quantization: str = "8bit",
        torch_dtype_str: str = "float16",
        use_gpu: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_gpu = use_gpu

        # Resolve torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype_str, torch.float16)

        logger.info("Loading QARI model: %s", model_name)

        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2VLForConditionalGeneration,
        )

        # Quantization config
        if quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        else:
            bnb_config = None

        device_map = "auto" if use_gpu else "cpu"

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": device_map,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, use_fast=False,
        )

        logger.info("QARI model loaded successfully.")

    def ocr_page(
        self,
        pil_image: Image.Image,
        ocr_prompt: str,
        page_number: int = 1,
    ) -> dict:
        """Run OCR on a single page image.

        Parameters
        ----------
        pil_image:
            The page image to process.
        ocr_prompt:
            The prompt to send to the model.
        page_number:
            1-based page number (for logging).

        Returns
        -------
        dict
            Keys: ``raw_text``, ``confidence``, ``word_confidences``, ``error``.
        """
        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": ocr_prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            generated_ids = outputs.sequences
            scores = outputs.scores  # tuple of logit tensors

            # Trim input tokens from generated sequence
            input_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[0, input_len:]

            # Decode text
            raw_text = self.processor.batch_decode(
                [generated_ids_trimmed],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Compute confidence
            page_conf = compute_page_confidence(scores, generated_ids_trimmed)

            # Attempt per-word confidence (stretch goal)
            word_confs = compute_word_confidences(
                scores, generated_ids_trimmed, self.processor.tokenizer,
            )

            return {
                "raw_text": raw_text,
                "confidence": round(page_conf, 4),
                "word_confidences": word_confs,
                "error": None,
            }

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM on page %d -- skipping.", page_number)
            torch.cuda.empty_cache()
            return {
                "raw_text": "",
                "confidence": None,
                "word_confidences": None,
                "error": f"CUDA OOM -- page {page_number} skipped",
            }
        except Exception as exc:
            logger.exception("OCR engine error on page %d: %s", page_number, exc)
            return {
                "raw_text": "",
                "confidence": None,
                "word_confidences": None,
                "error": str(exc),
            }


def get_engine(
    model_name: str = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
    max_new_tokens: int = 4000,
    quantization: str = "8bit",
    torch_dtype_str: str = "float16",
    use_gpu: bool = True,
) -> QARIEngine:
    """Return the singleton ``QARIEngine`` instance, creating it if needed."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = QARIEngine(
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    quantization=quantization,
                    torch_dtype_str=torch_dtype_str,
                    use_gpu=use_gpu,
                )
    return _engine_instance
