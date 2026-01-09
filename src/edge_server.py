import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

import numpy as np
import requests
from flask import Flask, request, jsonify, Response

import torch
import torch.nn.functional as F
from torchvision import transforms

from mvit_backbone import mobilevit_xxs
from mvit_ee_paper import MobileViTMultiExitLogits
from criterions import score_entropy, score_margin, score_maxprob
from payloads import decode_u8_image_bytes, u8_to_pil_rgb

import msgpack

SESSION = requests.Session()
app = Flask(__name__)

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE_NAME = "edge_requests.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_log_path(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, LOG_FILE_NAME)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def parse_payload() -> Tuple[Dict[str, Any], str]:
    """
    Supports:
      - application/json
      - application/msgpack (or application/x-msgpack)
    """
    ct = (request.content_type or "").lower()

    if "application/json" in ct:
        data = request.get_json(force=True, silent=False)
        return data if isinstance(data, dict) else {}, "json"

    if ("application/msgpack" in ct or "application/x-msgpack" in ct):
        raw = request.get_data()
        data = msgpack.unpackb(raw, raw=False)
        return data if isinstance(data, dict) else {}, "msgpack"

    # Fallback: try JSON
    data = request.get_json(force=True, silent=True)
    if isinstance(data, dict):
        return data, "json(fallback)"

    return {}, "unknown"


def pack_response(resp: Dict[str, Any], accept: str) -> Response:
    """
    If client Accept includes msgpack -> return msgpack.
    Else JSON.
    """
    accept = (accept or "").lower()
    if "application/msgpack" in accept or "application/x-msgpack" in accept:
        body = msgpack.packb(resp, use_bin_type=True)
        return Response(body, content_type="application/msgpack")
    return jsonify(resp)


def build_infer_transform():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ])


def load_model(device: torch.device, backbone_ckpt: str, ee_ckpt: str, num_classes: int = 10):
    base = mobilevit_xxs()
    base.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))

    multi = MobileViTMultiExitLogits(
        base_model=base,
        exit_points=("mvit_0", "mvit_1"),
        num_classes=num_classes
    )
    multi.load_state_dict(torch.load(ee_ckpt, map_location="cpu"), strict=False)
    multi = multi.to(device).eval()
    return multi

def choose_exit_or_offload(outputs: dict, criterion: str, thr0: float, thr1: float):
    z0 = outputs["logits_mvit_0"]
    z1 = outputs["logits_mvit_1"]

    if criterion == "maxprob":
        s0 = score_maxprob(z0)
        s1 = score_maxprob(z1)
        pass0 = s0 >= thr0
        pass1 = s1 >= thr1
    elif criterion == "margin":
        s0 = score_margin(z0)
        s1 = score_margin(z1)
        pass0 = s0 >= thr0
        pass1 = s1 >= thr1
    elif criterion == "entropy":
        s0 = score_entropy(z0)
        s1 = score_entropy(z1)
        pass0 = s0 <= thr0
        pass1 = s1 <= thr1
    else:
        raise ValueError("criterion must be one of: maxprob, margin, entropy")

    if pass0.item():
        return "exit0", float(s0.item()), float(s1.item())
    if pass1.item():
        return "exit1", float(s0.item()), float(s1.item())
    return "offload", float(s0.item()), float(s1.item())


def dummy_offload_placeholder(outputs: dict) -> torch.Tensor:
    """
    Placeholder OFFLOAD behavior (no networking yet):
    - for debugging, we use logits_final computed on edge.
    Later: send features to cloud and receive logits.
    """
    return outputs["logits_final"]


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "edge",
        "time_utc": now_iso(),
        "cloud_url": app.config.get("CLOUD_URL"),
        "device": str(app.config.get("DEVICE")),
    })


@app.post("/infer")
def infer():
    """
    Proper inference endpoint.

    Recommended request (msgpack):
      {
        "sample_id": int,
        "image_u8": <bytes>,        # raw bytes length 32*32*3
        "shape": [32,32,3],         # optional, defaults CIFAR
        "label": int,               # optional (for accuracy)
        "policy": {
            "criterion": "maxprob"|"entropy"|"margin",
            "thr0": float,
            "thr1": float
        }
      }

    JSON alternative:
      If JSON, you cannot send raw bytes easily; use base64 if you want JSON.
      For now, use msgpack for real image inference.
    """
    req_received_utc = now_iso()
    t_total0 = time.perf_counter()

    try:
        payload, used_ct = parse_payload()
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    sample_id = payload.get("sample_id", None)

    # --- Decode image ---
    try:
        shape = payload.get("shape", [32, 32, 3])
        shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        img_bytes = payload.get("image_u8", None)

        if img_bytes is None:
            raise ValueError("Missing field 'image_u8' (raw bytes). Use msgpack payload.")
        if not isinstance(img_bytes, (bytes, bytearray)):
            raise ValueError(f"'image_u8' must be bytes, got {type(img_bytes)}")

        img_u8 = decode_u8_image_bytes(bytes(img_bytes), shape_hw3=shape)
        pil = u8_to_pil_rgb(img_u8)
    except Exception as e:
        return jsonify({"error": "bad_image", "message": str(e)}), 400

    # --- Preprocess ---
    tfm = app.config["INFER_TFM"]
    x = tfm(pil).unsqueeze(0)  # (1,3,256,256)

    device = app.config["DEVICE"]
    model = app.config["MODEL"]
    x = x.to(device)

    # --- Policy ---
    policy = payload.get("policy", {}) or {}
    criterion = policy.get("criterion", "maxprob")
    thr0 = float(policy.get("thr0", 0.90))
    thr1 = float(policy.get("thr1", 0.95))

    # --- Inference timing (GPU sync matters) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        outputs = model(x)
        decision, s0, s1 = choose_exit_or_offload(outputs, criterion, thr0, thr1)

        if decision == "exit0":
            logits = outputs["logits_mvit_0"]
        elif decision == "exit1":
            logits = outputs["logits_mvit_1"]
        else:
            # OFFLOAD placeholder: use final logits locally
            logits = dummy_offload_placeholder(outputs)

        pred = int(logits.argmax(dim=1).item())

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    edge_compute_ms = (t1 - t0) * 1000.0

    # Placeholder offload info (no cloud call yet)
    edge_cloud_rtt_ms = None
    cloud_error = None

    correct = None

    t_total1 = time.perf_counter()
    total_ms = (t_total1 - t_total0) * 1000.0

    resp = {
        "sample_id": sample_id,
        "decision": decision,
        "offload": (decision == "offload"),
        "pred": pred,
        "correct": correct,

        "criterion": criterion,
        "thr0": thr0,
        "thr1": thr1,
        "score_exit0": s0,
        "score_exit1": s1,

        "edge_compute_ms": edge_compute_ms,
        "edge_cloud_rtt_ms": edge_cloud_rtt_ms,
        "total_edge_ms": total_ms,

        "cloud_error": cloud_error,
        "received_content_type": used_ct,
        "time_utc": now_iso(),
        "note": "edge runs real model; OFFLOAD is placeholder using logits_final locally",
    }

    # Log compact record
    log_dir = app.config.get("LOG_DIR", DEFAULT_LOG_DIR)
    log_path = ensure_log_path(log_dir)
    append_jsonl(log_path, {
        "time_utc": resp["time_utc"],
        "req_received_utc": req_received_utc,
        "sample_id": sample_id,
        "decision": decision,
        "offload": resp["offload"],
        "criterion": criterion,
        "thr0": thr0,
        "thr1": thr1,
        "score_exit0": s0,
        "score_exit1": s1,
        "edge_compute_ms": edge_compute_ms,
        "total_edge_ms": total_ms,
    })

    # return msgpack if client asks for it
    return pack_response(resp, request.headers.get("Accept", ""))

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)

    ap.add_argument("--cloud_url", type=str, required=False, default=None,
                    help="Cloud base URL (unused for now; offload is placeholder)")
    ap.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)

    ap.add_argument("--backbone_ckpt", type=str, required=True)
    ap.add_argument("--ee_ckpt", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=10)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, args.backbone_ckpt, args.ee_ckpt, num_classes=args.num_classes)
    infer_tfm = build_infer_transform()

    app.config["CLOUD_URL"] = args.cloud_url
    app.config["LOG_DIR"] = args.log_dir
    app.config["DEVICE"] = device
    app.config["MODEL"] = model
    app.config["INFER_TFM"] = infer_tfm

    print(f"[edge] starting on http://{args.host}:{args.port}")
    print(f"[edge] device: {device}")
    print(f"[edge] backbone_ckpt: {args.backbone_ckpt}")
    print(f"[edge] ee_ckpt: {args.ee_ckpt}")
    print("[edge] /infer expects msgpack payload with raw CIFAR image bytes (image_u8)")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()