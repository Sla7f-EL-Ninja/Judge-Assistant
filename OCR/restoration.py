"""
OCR.restoration
---------------
Image restoration: resize and CLAHE contrast normalization.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def restore_image(
    pil_image: Image.Image,
    max_image_dimension: int = 4000,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
) -> Image.Image:
    """Lightweight image restoration.

    Steps:
    1. Convert to RGB if needed.
    2. Resize if largest dimension exceeds ``max_image_dimension``.
    3. Apply CLAHE contrast normalization on the L channel of LAB colour space.

    Parameters
    ----------
    pil_image:
        Input PIL image.
    max_image_dimension:
        Maximum allowed dimension (width or height) in pixels.
    clahe_clip_limit:
        CLAHE clip limit parameter.
    clahe_tile_grid_size:
        CLAHE tile grid size.

    Returns
    -------
    PIL.Image.Image
        The restored image.
    """
    # 1. RGB conversion
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # 2. Resize if too large
    w, h = pil_image.size
    max_dim = max(w, h)
    if max_dim > max_image_dimension:
        scale = max_image_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        logger.info("Resized: %dx%d -> %dx%d", w, h, new_w, new_h)

    # 3. CLAHE contrast normalization
    img_array = np.array(pil_image)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size,
    )
    l_enhanced = clahe.apply(l_ch)
    merged = cv2.merge((l_enhanced, a_ch, b_ch))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced_rgb)
