"""
preprocessor.py

Image preprocessing pipeline to maximize OCR accuracy
on Arabic legal documents.

Surya handles many aspects internally, so preprocessing is lighter
than with older engines like EasyOCR. Each step is independently
toggleable via config. The pipeline works with PIL Images throughout,
converting to OpenCV format only for specific operations.
"""

import logging
import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import config.ocr as config

logger = logging.getLogger(__name__)


def preprocess_image(
    image: Image.Image,
    enable_denoise: Optional[bool] = None,
    enable_deskew: Optional[bool] = None,
    enable_border_removal: Optional[bool] = None,
    enable_contrast_enhancement: Optional[bool] = None,
    enable_resolution_check: Optional[bool] = None,
) -> Image.Image:
    """
    Run the preprocessing pipeline on a single PIL Image.

    Surya works well with RGB input, so we avoid aggressive binarization.
    The pipeline focuses on resolution, deskewing, border removal, and
    contrast enhancement.

    Args:
        image: Input PIL Image in RGB mode.
        enable_denoise: Override config ENABLE_DENOISE.
        enable_deskew: Override config ENABLE_DESKEW.
        enable_border_removal: Override config ENABLE_BORDER_REMOVAL.
        enable_contrast_enhancement: Override config ENABLE_CONTRAST_ENHANCEMENT.
        enable_resolution_check: Override config ENABLE_RESOLUTION_CHECK.

    Returns:
        Preprocessed PIL Image in RGB mode, ready for Surya.
    """
    if enable_denoise is None:
        enable_denoise = config.ENABLE_DENOISE
    if enable_deskew is None:
        enable_deskew = config.ENABLE_DESKEW
    if enable_border_removal is None:
        enable_border_removal = config.ENABLE_BORDER_REMOVAL
    if enable_contrast_enhancement is None:
        enable_contrast_enhancement = config.ENABLE_CONTRAST_ENHANCEMENT
    if enable_resolution_check is None:
        enable_resolution_check = config.ENABLE_RESOLUTION_CHECK

    result = image.copy()

    # 1. Resolution check — upscale low-res images
    if enable_resolution_check:
        result = check_and_upscale_resolution(result)
        logger.debug("Resolution check done")

    # 2. Deskewing — correct rotation
    if enable_deskew:
        result = deskew(result)
        logger.debug("Deskewing done")

    # 3. Border removal — crop black scan borders
    if enable_border_removal:
        result = remove_borders(result)
        logger.debug("Border removal done")

    # 4. Contrast enhancement — CLAHE for faded documents
    if enable_contrast_enhancement:
        result = enhance_contrast(result)
        logger.debug("Contrast enhancement done")

    # 5. Noise removal — optional, off by default since Surya is robust
    if enable_denoise:
        result = denoise(result)
        logger.debug("Denoising done")

    return result


def check_and_upscale_resolution(image: Image.Image) -> Image.Image:
    """
    Ensure minimum resolution for OCR accuracy.

    If the image height is below what a standard A4 page would be
    at MIN_DPI, upscale using Lanczos interpolation.
    """
    min_dpi = config.MIN_DPI

    # A4 page at min DPI: 297mm * (dpi / 25.4)
    expected_height_at_min = int(min_dpi * 11.69)

    w, h = image.size

    if h >= expected_height_at_min:
        return image

    scale = expected_height_at_min / h
    # Cap upscaling at 3x to avoid excessive memory use
    scale = min(scale, 3.0)

    if scale <= 1.05:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    logger.info("Upscaling image from %dx%d to %dx%d (%.1fx)", w, h, new_w, new_h, scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def deskew(image: Image.Image) -> Image.Image:
    """
    Detect and correct document rotation using Hough line transform.

    Handles small rotations typical of scanned documents (up to ~15 degrees).
    Surya handles minor skew but large rotations need correction.
    """
    # Convert to grayscale numpy array for OpenCV processing
    gray = np.array(image.convert("L"))

    coords = np.column_stack(np.where(gray < 128))

    if len(coords) < 50:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angles in [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct if angle is small (large angles likely mean
    # the document is intentionally rotated, not skewed)
    if abs(angle) > 15 or abs(angle) < 0.1:
        return image

    logger.info("Deskewing by %.2f degrees", angle)
    return image.rotate(-angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def remove_borders(image: Image.Image) -> Image.Image:
    """
    Remove black scan borders by finding the largest contour
    that represents the document area.
    """
    gray = np.array(image.convert("L"))

    # Invert so document area is white
    inverted = cv2.bitwise_not(gray)

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    img_area = gray.shape[0] * gray.shape[1]
    contour_area = w * h
    if contour_area < img_area * 0.5:
        return image

    # Add small padding
    pad = 5
    y_start = max(0, y - pad)
    y_end = min(gray.shape[0], y + h + pad)
    x_start = max(0, x - pad)
    x_end = min(gray.shape[1], x + w + pad)

    return image.crop((x_start, y_start, x_end, y_end))


def enhance_contrast(image: Image.Image) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve contrast on faded documents.

    Works on the L channel in LAB color space to preserve color information.
    """
    arr = np.array(image)

    # Convert RGB to LAB
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)


def denoise(image: Image.Image) -> Image.Image:
    """
    Remove noise using Non-Local Means denoising.

    Optional step — off by default since Surya is robust to noise.
    """
    arr = np.array(image)

    # Use color denoising for RGB images (positional args for OpenCV compat)
    denoised = cv2.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)

    return Image.fromarray(denoised)
