import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

from flask import Flask, request, jsonify, Response
import msgpack
import torch
from codecarbon import EmissionsTracker
import threading

from mvit_backbone import mobilevit_xxs
from mvit_cloud import MobileViTCloudContinuation
from payloads import payload_to_tensor, tensor_to_payload

app = Flask(__name__)

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE_NAME = "cloud_requests.jsonl"

ENERGY_LOG_FILE_NAME = "cloud_energy.jsonl"
_TRACKERS_LOCK = threading.Lock()
_ACTIVE_TRACKERS: Dict[str, EmissionsTracker] = {}


def ensure_log_path(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, LOG_FILE_NAME)


def ensure_energy_log_path(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, ENERGY_LOG_FILE_NAME)


def _make_tracker(log_dir: str, run_id: str) -> EmissionsTracker:
    """
    Create a CodeCarbon tracker for a sweep (run_id).
    We deliberately keep it lightweight and file-output-free
    because we log totals ourselves to JSONL.
    """
    return EmissionsTracker(
        project_name="edge_cloud_early_exit",
        experiment_id=run_id,
        output_dir=log_dir,
        save_to_file=False,     # we log ourselves
        log_level="error",      # reduce noise
    )


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


@app.post("/energy/start")
def energy_start_endpoint():
    """
    Starts an EmissionsTracker for a given run_id (one sweep).
    Body: {"run_id": "..."}  (msgpack/json)
    """
    req_received_utc = now_iso()

    try:
        payload, used_ct = parse_payload()
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    run_id = payload.get("run_id", None)
    if not run_id or not isinstance(run_id, str):
        return jsonify({"error": "bad_request", "message": "Missing/invalid 'run_id' (string)"}), 400

    log_dir = app.config.get("LOG_DIR", DEFAULT_LOG_DIR)

    with _TRACKERS_LOCK:
        if run_id in _ACTIVE_TRACKERS:
            # already started: idempotent success
            resp = {
                "status": "ok",
                "service": app.config.get("SERVICE_NAME", "unknown"),
                "run_id": run_id,
                "started": False,
                "message": "tracker already running",
                "time_utc": now_iso(),
                "received_content_type": used_ct,
            }
            return pack_response(resp, request.headers.get("Accept", ""))

        tracker = _make_tracker(log_dir=log_dir, run_id=run_id)
        tracker.start()
        _ACTIVE_TRACKERS[run_id] = tracker

    resp = {
        "status": "ok",
        "service": app.config.get("SERVICE_NAME", "unknown"),
        "run_id": run_id,
        "started": True,
        "time_utc": now_iso(),
        "received_content_type": used_ct,
    }

    # Optional: log start event
    energy_log_path = ensure_energy_log_path(log_dir)
    append_jsonl(energy_log_path, {
        "event": "energy_start",
        "time_utc": resp["time_utc"],
        "req_received_utc": req_received_utc,
        "run_id": run_id,
    })

    return pack_response(resp, request.headers.get("Accept", ""))


@app.post("/energy/stop")
def energy_stop_endpoint():
    """
    Stops a tracker for run_id and returns totals.
    Body: {"run_id": "..."} (msgpack/json)
    """
    req_received_utc = now_iso()

    try:
        payload, used_ct = parse_payload()
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    run_id = payload.get("run_id", None)
    if not run_id or not isinstance(run_id, str):
        return jsonify({"error": "bad_request", "message": "Missing/invalid 'run_id' (string)"}), 400

    with _TRACKERS_LOCK:
        tracker = _ACTIVE_TRACKERS.pop(run_id, None)

    if tracker is None:
        return jsonify({
            "error": "not_found",
            "message": f"No active tracker for run_id={run_id}. Did you call /energy/start?",
        }), 404

    # CodeCarbon returns emissions (kg CO2e) from stop()
    try:
        tracker.stop()
    except Exception as e:
        return jsonify({"error": "tracker_stop_failed", "message": str(e)}), 500

    # Prefer notebook-style extraction
    co2_kg = None
    energy_kwh = None
    duration_s = None

    try:
        data = tracker._prepare_emissions_data()  # internal but stable in your baseline
        co2_kg = getattr(data, "emissions", None)
        energy_kwh = getattr(data, "energy_consumed", None)
        duration_s = getattr(data, "duration", None)
    except Exception:
        # Fallbacks across versions
        print("Warning: failed to extract emissions data via _prepare_emissions_data()")
        try:
            fed = getattr(tracker, "final_emissions_data", None)
            if fed is not None:
                co2_kg = co2_kg if co2_kg is not None else getattr(fed, "emissions", None)
                energy_kwh = energy_kwh if energy_kwh is not None else getattr(fed, "energy_consumed", None)
                duration_s = duration_s if duration_s is not None else getattr(fed, "duration", None)
        except Exception:
            pass

    resp = {
        "status": "ok",
        "service": app.config.get("SERVICE_NAME", "unknown"),
        "run_id": run_id,
        "co2_kg": float(co2_kg) if co2_kg is not None else None,
        "energy_kwh": float(energy_kwh) if energy_kwh is not None else None,
        "duration_s": float(duration_s) if duration_s is not None else None,
        "time_utc": now_iso(),
        "received_content_type": used_ct,
    }

    # Log stop event + totals
    log_dir = app.config.get("LOG_DIR", DEFAULT_LOG_DIR)
    energy_log_path = ensure_energy_log_path(log_dir)
    append_jsonl(energy_log_path, {
        "event": "energy_stop",
        "time_utc": resp["time_utc"],
        "req_received_utc": req_received_utc,
        "run_id": run_id,
        "co2_kg": resp["co2_kg"],
        "energy_kwh": resp["energy_kwh"],
        "duration_s": resp["duration_s"],
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
    app.config["SERVICE_NAME"] = "cloud"

    print(f"[cloud] starting on http://{args.host}:{args.port}")
    print(f"[cloud] device: {device}")
    print(f"[cloud] backbone_ckpt: {args.backbone_ckpt}")
    print(f"[cloud] logs: {os.path.abspath(args.log_dir)}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
