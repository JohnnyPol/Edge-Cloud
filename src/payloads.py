# src/payloads.py
import numpy as np
from PIL import Image

def decode_u8_image_bytes(img_bytes: bytes, shape_hw3=(32, 32, 3)) -> np.ndarray:
    """
    img_bytes: raw bytes length H*W*3
    Returns: np.uint8 array (H,W,3)
    """
    h, w, c = shape_hw3
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    if arr.size != h * w * c:
        raise ValueError(f"Bad image size: got {arr.size}, expected {h*w*c}")
    return arr.reshape((h, w, c))

def u8_to_pil_rgb(img_u8_hwc: np.ndarray) -> Image.Image:
    if img_u8_hwc.dtype != np.uint8:
        raise ValueError("Expected uint8 image")
    if img_u8_hwc.ndim != 3 or img_u8_hwc.shape[2] != 3:
        raise ValueError("Expected HxWx3 image")
    return Image.fromarray(img_u8_hwc, mode="RGB")
