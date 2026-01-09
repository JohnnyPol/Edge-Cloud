import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

from flask import Flask, request, jsonify, Response
import msgpack
import torch

from mvit_backbone import mobilevit_xxs
from mvit_cloud import MobileViTCloudContinuation
from payloads import payload_to_tensor, tensor_to_payload

app = Flask(__name__)

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE_NAME = "cloud_requests.jsonl"


def ensure_log_path(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, LOG_FILE_NAME)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def parse_payload() -> Tuple[Dict[str, Any], str]:
    """
    Returns (payload_dict, content_type_used)
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
    accept = (accept or "").lower()
    if "application/msgpack" in accept or "application/x-msgpack" in accept:
        body = msgpack.packb(resp, use_bin_type=True)
        return Response(body, content_type="application/msgpack")
    return jsonify(resp)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "cloud",
        "time_utc": now_iso(),
        "device": str(app.config.get("DEVICE")),
    })


@app.get("/ping")
def ping():
    return jsonify({
        "status": "ok",
        "service": "cloud",
        "time_utc": now_iso(),
    })


@app.post("/continue")
def continue_inference():
    req_received_utc = now_iso()
    t_total0 = time.perf_counter()

    try:
        payload, used_ct = parse_payload()
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    sample_id = payload.get("sample_id", None)
    from_exit = payload.get("from_exit", None)

    feat_payload = payload.get("features", None)
    if feat_payload is None:
        return jsonify({"error": "bad_request", "message": "Missing 'features'"}), 400

    # Validate feature payload shape/dtype metadata
    try:
        dtype = feat_payload["dtype"]
        shape = feat_payload["shape"]
        data_len = len(feat_payload["data"])
    except Exception as e:
        return jsonify({"error": "bad_features", "message": str(e)}), 400

    device = app.config["DEVICE"]
    cont_model = app.config["CONT_MODEL"]

    # --- Cloud compute timing (model only) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        feat_tensor = payload_to_tensor(feat_payload, device=device)
        logits = cont_model.forward_from_exit(feat_tensor, from_exit=from_exit)
    except Exception as e:
        return jsonify({"error": "cloud_infer_failed", "message": str(e)}), 500

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    cloud_compute_ms = (t1 - t0) * 1000.0
    t_total1 = time.perf_counter()
    total_cloud_ms = (t_total1 - t_total0) * 1000.0

    logits_payload = tensor_to_payload(logits)

    resp = {
        "sample_id": sample_id,
        "from_exit": from_exit,
        "cloud_compute_ms": cloud_compute_ms,
        "total_cloud_ms": total_cloud_ms,
        "received_content_type": used_ct,
        "time_utc": now_iso(),
        "note": "cloud continuation executed",
        "logits": logits_payload,
        "features_meta": {
            "dtype": dtype,
            "shape": shape,
            "data_len": data_len
        }
    }

    # Log compact record
    log_dir = app.config.get("LOG_DIR", DEFAULT_LOG_DIR)
    log_path = ensure_log_path(log_dir)
    append_jsonl(log_path, {
        "time_utc": resp["time_utc"],
        "req_received_utc": req_received_utc,
        "endpoint": "/continue",
        "sample_id": sample_id,
        "from_exit": from_exit,
        "cloud_compute_ms": cloud_compute_ms,
        "total_cloud_ms": total_cloud_ms,
        "features_shape": shape,
        "features_dtype": dtype,
        "features_data_len": data_len,
    })

    return pack_response(resp, request.headers.get("Accept", ""))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    ap.add_argument("--debug", action="store_true")

    # Model load
    ap.add_argument("--backbone_ckpt", type=str, required=True,
                    help="Path to mobileViT_xxs backbone checkpoint (e.g. ../data/mobileViT_xxs_10.pth)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = mobilevit_xxs()
    base.load_state_dict(torch.load(args.backbone_ckpt, map_location="cpu"))
    base = base.to(device).eval()

    cont_model = MobileViTCloudContinuation(base).to(device).eval()

    app.config["DEVICE"] = device
    app.config["CONT_MODEL"] = cont_model
    app.config["LOG_DIR"] = args.log_dir

    print(f"[cloud] starting on http://{args.host}:{args.port}")
    print(f"[cloud] device: {device}")
    print(f"[cloud] backbone_ckpt: {args.backbone_ckpt}")
    print(f"[cloud] logs: {os.path.abspath(args.log_dir)}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
