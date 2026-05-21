"""
DocumentProcessor.OCR.ocr_engine
----------------------------------
QARI OCR engine: model loading and per-page inference.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import torch
from PIL import Image

from DocumentProcessor.OCR.confidence import compute_page_confidence, compute_word_confidences
from DocumentProcessor.OCR.models import WordConfidence

logger = logging.getLogger(__name__)

_engine_instance: Optional["QARIEngine"] = None
_engine_lock = threading.Lock()

# Errors that indicate the quantized model won't fit on GPU and we should
# fall back to a full-precision CPU run instead of crashing.
_GPU_OFFLOAD_ERRORS = (
    "dispatched on the CPU or the disk",
    "llm_int8_enable_fp32_cpu_offload",
)


def _build_bnb_config(quantization: str):
    """Return a BitsAndBytesConfig for the requested quantization, or None."""
    from transformers import BitsAndBytesConfig

    if quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    return None


class QARIEngine:
    """Wrapper around the QARI (Qwen2VL-based) OCR model."""

    def __init__(
        self,
        model_name: str = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
        max_new_tokens: int = 4000,
        quantization: str = "4bit",
        torch_dtype_str: str = "float16",
        use_gpu: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_gpu = use_gpu

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype_str, torch.float16)

        logger.info("Loading QARI model: %s", model_name)

        from transformers import (
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
        )

        bnb_config = _build_bnb_config(quantization) if use_gpu else None
        device_map = "auto" if use_gpu else "cpu"

        load_kwargs: dict = {
            "dtype": self.torch_dtype,   # replaces deprecated torch_dtype
            "device_map": device_map,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        # --- Attempt 1: GPU + quantization (or plain GPU) ---
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, **load_kwargs,
            )
        except (ValueError, RuntimeError) as exc:
            exc_str = str(exc)
            if use_gpu and any(marker in exc_str for marker in _GPU_OFFLOAD_ERRORS):
                logger.warning(
                    "GPU VRAM too small for quantized model (%s). "
                    "Falling back to full-precision CPU inference.",
                    exc,
                )
                # --- Attempt 2: CPU fallback, no quantization ---
                self.use_gpu = False
                cpu_kwargs: dict = {
                    "dtype": torch.float32,
                    "device_map": "cpu",
                }
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name, **cpu_kwargs,
                )
                logger.info("Model loaded on CPU (float32, no quantization).")
            else:
                raise

        self.processor = AutoProcessor.from_pretrained(
            model_name, use_fast=False,
        )

        logger.info("QARI model loaded successfully (use_gpu=%s).", self.use_gpu)

    def ocr_page(
        self,
        pil_image: Image.Image,
        ocr_prompt: str,
        page_number: int = 1,
    ) -> dict:
        """Run OCR on a single page image."""
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

            # Use GPU only if it was successfully initialised.
            device = (
                "cuda"
                if self.use_gpu and torch.cuda.is_available()
                else "cpu"
            )

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
            scores = outputs.scores

            input_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[0, input_len:]

            raw_text = self.processor.batch_decode(
                [generated_ids_trimmed],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            page_conf = compute_page_confidence(scores, generated_ids_trimmed)

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
    quantization: str = "4bit",
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