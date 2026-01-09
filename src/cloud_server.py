import os
import json
import time
from datetime import datetime
from typing import Any, Dict, Tuple

from flask import Flask, request, jsonify, Response

# Optional msgpack support (recommended for tensor payloads later)
try:
    import msgpack
    HAS_MSGPACK = True
except Exception:
    HAS_MSGPACK = False

app = Flask(__name__)

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE_NAME = "cloud_requests.jsonl"


def ensure_log_dir(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, LOG_FILE_NAME)


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


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
        "msgpack_enabled": HAS_MSGPACK,
    })


@app.post("/continue")
def continue_inference():
    t0 = time.perf_counter()

    try:
        payload, used_ct = parse_payload()
    except Exception as e:
        return jsonify({
            "error": "bad_request",
            "message": str(e),
        }), 400

    # Minimal fields we expect (optional for now)
    sample_id = payload.get("sample_id", None)
    from_exit = payload.get("from_exit", None)

    # Placeholder “compute”
    # (Later: deserialize features, run model continuation)
    time.sleep(0.002)  # 2ms fake compute

    # Placeholder response
    pred = 0
    logits_final = None  # keep None until real model is added

    t1 = time.perf_counter()
    cloud_compute_ms = (t1 - t0) * 1000.0

    resp = {
        "sample_id": sample_id,
        "from_exit": from_exit,
        "pred": pred,
        "logits_final": logits_final,
        "cloud_compute_ms": cloud_compute_ms,
        "received_content_type": used_ct,
        "time_utc": now_iso(),
        "note": "placeholder cloud response (no model yet)",
    }

    # Log request/response summary (avoid logging huge payloads)
    log_dir = app.config.get("LOG_DIR", DEFAULT_LOG_DIR)
    log_path = ensure_log_dir(log_dir)
    log_record = {
        "time_utc": resp["time_utc"],
        "endpoint": "/continue",
        "sample_id": sample_id,
        "from_exit": from_exit,
        "received_content_type": used_ct,
        "cloud_compute_ms": cloud_compute_ms,
        "payload_keys": sorted(list(payload.keys())) if isinstance(payload, dict) else [],
    }
    append_jsonl(log_path, log_record)

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
    print(f"[cloud] msgpack enabled: {HAS_MSGPACK}")

    # NOTE: Flask dev server is fine for your prototype.
    # Later you can switch to gunicorn for robustness.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
