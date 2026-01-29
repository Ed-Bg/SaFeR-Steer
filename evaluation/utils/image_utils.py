"""
Image processing utilities for evaluation
"""

import math
import base64
from io import BytesIO
from PIL import Image


def check_and_resize_image(
    image_path: str,
    max_pixels: int = 512 * 512,
    min_pixels: int = 338 * 338
) -> Image.Image:
    """
    Load and resize image to fit within pixel bounds.
    
    Args:
        image_path: Path to the image file
        max_pixels: Maximum total pixels allowed
        min_pixels: Minimum total pixels required
    
    Returns:
        Resized PIL Image in RGB mode
    """
    image = Image.open(image_path).convert("RGB")
    image.load()
    
    current_pixels = image.width * image.height
    
    if current_pixels > max_pixels:
        resize_factor = math.sqrt(max_pixels / current_pixels)
        new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
        image = image.resize(new_size)
    
    if current_pixels < min_pixels:
        resize_factor = math.sqrt(min_pixels / current_pixels)
        new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
        image = image.resize(new_size)
    
    return image


def encode_image(image_path: str, max_pixels: int = 512 * 512, min_pixels: int = 338 * 338) -> str:
    """
    Load, resize and encode image to base64.
    
    Args:
        image_path: Path to the image file
        max_pixels: Maximum total pixels allowed
        min_pixels: Minimum total pixels required
    
    Returns:
        Base64 encoded image string
    """
    img = check_and_resize_image(image_path, max_pixels, min_pixels)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
