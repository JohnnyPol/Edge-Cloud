# src/payloads.py
import numpy as np
import torch
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

def tensor_to_payload(t: torch.Tensor) -> dict:
    """
    Converts a torch tensor to a msgpack-friendly dict:
      - dtype (string)
      - shape (list)
      - data (bytes)
    Always sends CPU contiguous bytes.
    """
    t_cpu = t.detach().to("cpu").contiguous()
    arr = t_cpu.numpy()  # shares memory with t_cpu
    return {
        "dtype": str(arr.dtype),           # e.g. 'float32'
        "shape": list(arr.shape),          # e.g. [1, 64, 16, 16]
        "data": arr.tobytes(order="C"),    # raw bytes
    }

def payload_to_tensor(p: dict, device: str = "cpu") -> torch.Tensor:
    """
    Reconstruct torch tensor from dict produced by tensor_to_payload.
    """
    dtype = np.dtype(p["dtype"])
    shape = tuple(p["shape"])
    data = p["data"]
    arr = np.frombuffer(data, dtype=dtype).reshape(shape)
    t = torch.from_numpy(arr)
    return t.to(device)