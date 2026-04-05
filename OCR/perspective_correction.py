"""
OCR.perspective_correction
--------------------------
Perspective correction using edge-based contour detection with
adaptive-threshold fallback.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest sum
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right: smallest diff
    rect[3] = pts[np.argmax(d)]   # bottom-left: largest diff
    return rect


def expand_quad(
    pts: np.ndarray,
    img_shape: Tuple[int, ...],
    margin_pct: float = 0.02,
) -> np.ndarray:
    """Expand a quadrilateral outward from its centre by *margin_pct*
    to avoid clipping text at the very edge of the page."""
    h, w = img_shape[:2]
    center = pts.mean(axis=0)
    expanded = center + (pts - center) * (1 + margin_pct)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return expanded.astype(np.float32)


# ---------------------------------------------------------------------------
# Approach 1 -- Edge-based contour detection
# ---------------------------------------------------------------------------

def detect_page_contour(
    img_array: np.ndarray,
    *,
    min_area_ratio: float = 0.35,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: Tuple[int, int] = (5, 5),
    dilate_kernel: Tuple[int, int] = (3, 3),
    dilate_iterations: int = 1,
    top_n_contours: int = 5,
    approx_epsilon: float = 0.02,
) -> Optional[np.ndarray]:
    """Edge-based contour detection.

    Finds the largest quadrilateral contour covering at least
    *min_area_ratio* of the image area.

    Returns 4 corner points or ``None``.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    h, w = img_array.shape[:2]
    img_area = w * h
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:top_n_contours]:
        area = cv2.contourArea(contour)
        if area < img_area * min_area_ratio:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_epsilon * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


# ---------------------------------------------------------------------------
# Approach 2 -- Adaptive threshold fallback
# ---------------------------------------------------------------------------

def detect_page_threshold(
    img_array: np.ndarray,
    *,
    min_area_ratio: float = 0.35,
    block_size: int = 35,
    c_constant: int = -10,
    close_kernel: Tuple[int, int] = (30, 30),
    close_iterations: int = 2,
    open_kernel: Tuple[int, int] = (15, 15),
    open_iterations: int = 1,
    approx_epsilon: float = 0.02,
) -> Optional[np.ndarray]:
    """Adaptive threshold fallback.

    Segments the light page from the darker background, then finds the
    largest connected component and returns its bounding quad.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    img_area = w * h

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_constant,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations,
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel_open, iterations=open_iterations,
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < img_area * min_area_ratio:
        return None

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, approx_epsilon * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2)

    # Fall back to minAreaRect
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

def _check_safety_guards(
    src_pts: np.ndarray,
    dst_w: int,
    dst_h: int,
    orig_w: int,
    orig_h: int,
    min_dim_ratio: float = 0.65,
    min_area_ratio: float = 0.35,
) -> Tuple[bool, str]:
    """Validate that the perspective correction is not too aggressive.

    Returns ``(is_safe, reason)``.
    """
    if dst_w < 100 or dst_h < 100:
        return False, f"output too small ({dst_w}x{dst_h})"

    w_ratio = dst_w / orig_w
    h_ratio = dst_h / orig_h
    if w_ratio < min_dim_ratio or h_ratio < min_dim_ratio:
        return False, (
            f"crops too aggressively (w_ratio={w_ratio:.2f}, h_ratio={h_ratio:.2f})"
        )

    area_ratio = (dst_w * dst_h) / (orig_w * orig_h)
    if area_ratio < min_area_ratio:
        return False, f"output area too small ({area_ratio:.2f} of original)"

    orig_portrait = orig_h > orig_w
    out_portrait = dst_h > dst_w
    if orig_portrait and not out_portrait:
        if dst_w / dst_h > 1.15:
            return False, "portrait original became landscape output"

    return True, "ok"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def perspective_correct(
    pil_image: Image.Image,
    *,
    # Contour detection params
    min_area_ratio: float = 0.35,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: Tuple[int, int] = (5, 5),
    dilate_kernel: Tuple[int, int] = (3, 3),
    dilate_iterations: int = 1,
    top_n_contours: int = 5,
    approx_epsilon_cd: float = 0.02,
    expand_margin_cd: float = 0.02,
    # Adaptive threshold params
    block_size: int = 35,
    c_constant: int = -10,
    close_kernel: Tuple[int, int] = (30, 30),
    close_iterations: int = 2,
    open_kernel: Tuple[int, int] = (15, 15),
    open_iterations: int = 1,
    approx_epsilon_at: float = 0.02,
    expand_margin_at: float = 0.02,
) -> Tuple[Image.Image, bool]:
    """Detect the physical page boundary and dewarp the image.

    Uses a tiered fallback:
      1. Edge-based contour detection
      2. Adaptive threshold page mask
      3. Passthrough (return original unchanged)

    Returns the corrected PIL Image and a boolean indicating whether
    correction was applied.
    """
    img_array = np.array(pil_image)
    h, w = img_array.shape[:2]

    try:
        # Approach 1: edge-based
        page_quad = detect_page_contour(
            img_array,
            min_area_ratio=min_area_ratio,
            canny_low=canny_low,
            canny_high=canny_high,
            blur_kernel=blur_kernel,
            dilate_kernel=dilate_kernel,
            dilate_iterations=dilate_iterations,
            top_n_contours=top_n_contours,
            approx_epsilon=approx_epsilon_cd,
        )
        method = "edge-contour"

        # Approach 2: adaptive threshold
        if page_quad is None:
            page_quad = detect_page_threshold(
                img_array,
                min_area_ratio=min_area_ratio,
                block_size=block_size,
                c_constant=c_constant,
                close_kernel=close_kernel,
                close_iterations=close_iterations,
                open_kernel=open_kernel,
                open_iterations=open_iterations,
                approx_epsilon=approx_epsilon_at,
            )
            method = "adaptive-threshold"

        # Approach 3: passthrough
        if page_quad is None:
            logger.info("No page boundary detected -- passing through.")
            return pil_image, False

        logger.info("Page boundary detected via %s", method)

        margin = expand_margin_cd if method == "edge-contour" else expand_margin_at
        page_quad = expand_quad(
            page_quad.astype(np.float32), img_array.shape, margin_pct=margin,
        )

        src_pts = order_points(page_quad)

        dst_w = int(max(
            np.linalg.norm(src_pts[1] - src_pts[0]),
            np.linalg.norm(src_pts[2] - src_pts[3]),
        ))
        dst_h = int(max(
            np.linalg.norm(src_pts[3] - src_pts[0]),
            np.linalg.norm(src_pts[2] - src_pts[1]),
        ))

        is_safe, reason = _check_safety_guards(
            src_pts, dst_w, dst_h, w, h, min_area_ratio=min_area_ratio,
        )
        if not is_safe:
            logger.info("Safety guard triggered: %s -- passing through.", reason)
            return pil_image, False

        dst_pts = np.array([
            [0,         0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0,         dst_h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            img_array, M, (dst_w, dst_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        logger.info(
            "Perspective corrected (%s): %dx%d -> %dx%d",
            method, w, h, dst_w, dst_h,
        )
        return Image.fromarray(warped), True

    except Exception as exc:
        logger.warning(
            "Perspective correction failed: %s -- passing through original.", exc,
        )
        return pil_image, False
