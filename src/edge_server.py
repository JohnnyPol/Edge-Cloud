import os
import json
import time
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import requests
from flask import Flask, request, jsonify

# Optional msgpack support (recommended later)
try:
    import msgpack
    HAS_MSGPACK = True
except Exception:
    HAS_MSGPACK = False

app = Flask(__name__)

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE_NAME = "edge_requests.jsonl"

SESSION = requests.Session()

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


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


def should_offload(payload: Dict[str, Any]) -> bool:
    """
    Placeholder controller:
    - If payload has "force_offload": true => offload
    - Else offload if "policy" says action == "offload"
    - Else default: no offload
    """
    if payload.get("force_offload") is True:
        return True
    policy = payload.get("policy", {})
    if isinstance(policy, dict) and policy.get("action") == "offload":
        return True
    return False

def call_cloud_ping(cloud_url: str, timeout_s: float = 5.0) -> Tuple[Optional[dict], Optional[float], Optional[str]]:
    url = cloud_url.rstrip("/") + "/ping"
    t0 = time.perf_counter()
    try:
        r = SESSION.get(url, timeout=timeout_s)
        t1 = time.perf_counter()
        rtt_ms = (t1 - t0) * 1000.0
        if r.status_code != 200:
            return None, rtt_ms, f"cloud_http_{r.status_code}: {r.text[:200]}"
        return r.json(), rtt_ms, None
    except Exception as e:
        t1 = time.perf_counter()
        return None, (t1 - t0) * 1000.0, str(e)


def call_cloud_continue(cloud_url: str, sample_id: Any, from_exit: str,
                        timeout_s: float = 5.0) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[str]]:
    """
    Calls cloud /continue and measures RTT.
    Returns: (cloud_response_json_or_none, rtt_ms_or_none, error_str_or_none)
    """
    url = cloud_url.rstrip("/") + "/continue"
    req = {
        "sample_id": sample_id,
        "from_exit": from_exit,
        # Placeholder: later we’ll attach features here.
        "features": None,
        "note": "placeholder offload from edge (no features yet)"
    }

    t0 = time.perf_counter()
    try:
        r = SESSION.post(url, json=req, timeout=timeout_s)
        t1 = time.perf_counter()
        rtt_ms = (t1 - t0) * 1000.0
        if r.status_code != 200:
            return None, rtt_ms, f"cloud_http_{r.status_code}: {r.text[:200]}"
        return r.json(), rtt_ms, None
    except Exception as e:
        t1 = time.perf_counter()
        rtt_ms = (t1 - t0) * 1000.0
        return None, rtt_ms, str(e)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "edge",
        "time_utc": now_iso(),
        "msgpack_enabled": HAS_MSGPACK,
        "cloud_url": app.config.get("CLOUD_URL"),
    })

@app.get("/cloud_ping")
def cloud_ping():
    cloud_url = app.config.get("CLOUD_URL")
    if not cloud_url:
        return jsonify({"error": "CLOUD_URL not configured"}), 400

    resp, rtt_ms, err = call_cloud_ping(
        cloud_url=cloud_url,
        timeout_s=app.config.get("CLOUD_TIMEOUT_S", 5.0)
    )
    return jsonify({
        "cloud_url": cloud_url,
        "ok": err is None,
        "edge_cloud_rtt_ms": rtt_ms,
        "cloud_response": resp,
        "error": err,
        "time_utc": now_iso(),
    })


@app.post("/infer")
def infer():
    """
    Placeholder inference endpoint.
    Expected payload fields (all optional for now):
      - sample_id
      - image (later)
      - label (optional)
      - force_offload (bool)
      - policy (dict)
    """
    req_received_utc = now_iso()
    t_total0 = time.perf_counter()

    try:
        payload, used_ct = parse_payload()
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    sample_id = payload.get("sample_id", None)
    label = payload.get("label", None)

    # Fake "edge compute"
    t_edge0 = time.perf_counter()
    time.sleep(0.003)  # 3ms placeholder
    # Placeholder local prediction
    local_pred = 0
    t_edge1 = time.perf_counter()
    edge_compute_ms = (t_edge1 - t_edge0) * 1000.0

    offload = should_offload(payload)
    decision = "exit_local" if not offload else "offload"

    cloud_resp = None
    edge_cloud_rtt_ms = None
    cloud_error = None
    final_pred = local_pred

    if offload:
        cloud_url = app.config.get("CLOUD_URL")
        if not cloud_url:
            cloud_error = "CLOUD_URL not configured on edge server"
        else:
            cloud_resp, edge_cloud_rtt_ms, cloud_error = call_cloud_continue(
                cloud_url=cloud_url,
                sample_id=sample_id,
                from_exit=payload.get("from_exit", "mvit_1"),
                timeout_s=app.config.get("CLOUD_TIMEOUT_S", 5.0)
            )
            if cloud_resp and ("pred" in cloud_resp):
                final_pred = cloud_resp["pred"]

    # correctness (if label provided)
    correct = None
    if label is not None:
        try:
            correct = int(final_pred) == int(label)
        except Exception:
            correct = None

    t_total1 = time.perf_counter()
    total_ms = (t_total1 - t_total0) * 1000.0

    resp = {
        "sample_id": sample_id,
        "decision": decision,
        "offload": bool(offload),
        "pred": final_pred,
        "label": label,
        "correct": correct,

        "edge_compute_ms": edge_compute_ms,
        "edge_cloud_rtt_ms": edge_cloud_rtt_ms,
        "total_edge_ms": total_ms,

        "cloud_response": cloud_resp,   # helpful for debugging
        "cloud_error": cloud_error,

        "received_content_type": used_ct,
        "time_utc": now_iso(),
        "note": "placeholder edge inference (no model yet)",
    }

    # Log a compact record
    log_dir = app.config.get("LOG_DIR", DEFAULT_LOG_DIR)
    log_path = ensure_log_path(log_dir)
    log_record = {
        "time_utc": resp["time_utc"],
        "req_received_utc": req_received_utc,
        "endpoint": "/infer",
        "sample_id": sample_id,
        "decision": decision,
        "offload": bool(offload),
        "edge_compute_ms": edge_compute_ms,
        "edge_cloud_rtt_ms": edge_cloud_rtt_ms,
        "total_edge_ms": total_ms,
        "cloud_error": cloud_error,
        "payload_keys": sorted(list(payload.keys())) if isinstance(payload, dict) else [],
    }
    append_jsonl(log_path, log_record)

    return jsonify(resp)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)

    ap.add_argument("--cloud_url", type=str, required=True,
                    help="Base URL of cloud server, e.g. http://CLOUD_IP:5002")
    ap.add_argument("--cloud_timeout_s", type=float, default=5.0)

    ap.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app.config["CLOUD_URL"] = args.cloud_url
    app.config["CLOUD_TIMEOUT_S"] = args.cloud_timeout_s
    app.config["LOG_DIR"] = args.log_dir

    print(f"[edge] starting on http://{args.host}:{args.port}")
    print(f"[edge] cloud url: {args.cloud_url}")
    print(f"[edge] logs: {os.path.abspath(args.log_dir)}")
    print(f"[edge] msgpack enabled: {HAS_MSGPACK}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
