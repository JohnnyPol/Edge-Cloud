import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from flask import Flask, request, jsonify, Response
import msgpack
import numpy as np


app = Flask(__name__)

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE_NAME = "cloud_requests.jsonl"


def ensure_log_dir(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, LOG_FILE_NAME)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_payload() -> Tuple[Dict[str, Any], str]:
    """
    Returns (payload_dict, content_type_used)
    Supports:
      - application/json
      - application/msgpack (or application/x-msgpack)
    """
    ct = (request.content_type or "").lower()

    # JSON
    if "application/json" in ct:
        data = request.get_json(force=True, silent=False)
        return data if isinstance(data, dict) else {}, "json"

    # msgpack
    if ("application/msgpack" in ct or "application/x-msgpack" in ct):
        if not HAS_MSGPACK:
            raise RuntimeError("msgpack not installed but msgpack payload received")
        raw = request.get_data()
        data = msgpack.unpackb(raw, raw=False)
        return data if isinstance(data, dict) else {}, "msgpack"

    # Fallback: try JSON
    data = request.get_json(force=True, silent=True)
    if isinstance(data, dict):
        return data, "json(fallback)"

    return {}, "unknown"


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "cloud",
        "time_utc": now_iso(),
    })


@app.post("/continue")
def continue_inference():
    t0 = time.perf_counter()

    try:
        payload, used_ct = parse_payload()
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    sample_id = payload.get("sample_id", None)
    from_exit = payload.get("from_exit", None)

    # Expect features in msgpack-friendly tensor payload format
    feat = payload.get("features", None)
    if feat is None:
        return jsonify({"error": "bad_request", "message": "Missing 'features'"}), 400

    # Minimal validation (we won't run model yet)
    try:
        dtype = feat["dtype"]
        shape = feat["shape"]
        data_len = len(feat["data"])
    except Exception as e:
        return jsonify({"error": "bad_features", "message": str(e)}), 400

    # Placeholder "cloud compute"
    time.sleep(0.002)

    # Create placeholder logits: (1,10) float32 all zeros except class 0
    logits = np.zeros((1, 10), dtype=np.float32)
    logits[0, 0] = 1.0

    t1 = time.perf_counter()
    cloud_compute_ms = (t1 - t0) * 1000.0

    resp = {
        "sample_id": sample_id,
        "from_exit": from_exit,
        "cloud_compute_ms": cloud_compute_ms,
        "received_content_type": used_ct,
        "time_utc": now_iso(),
        "note": "placeholder cloud continuation (features received, logits dummy)",
        # send logits as tensor payload too
        "logits": {
            "dtype": str(logits.dtype),
            "shape": list(logits.shape),
            "data": logits.tobytes(order="C")
        },
        "features_meta": {
            "dtype": dtype,
            "shape": shape,
            "data_len": data_len
        }
    }

    # If client asked for msgpack, return msgpack; else JSON
    accept = (request.headers.get("Accept") or "").lower()
    if "application/msgpack" in accept or "application/x-msgpack" in accept:
        body = msgpack.packb(resp, use_bin_type=True)
        return Response(body, content_type="application/msgpack")

    return jsonify(resp)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app.config["LOG_DIR"] = args.log_dir

    print(f"[cloud] starting on http://{args.host}:{args.port}")
    print(f"[cloud] logs: {os.path.abspath(args.log_dir)}")

    # NOTE: Flask dev server is fine for your prototype.
    # Later you can switch to gunicorn for robustness.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
