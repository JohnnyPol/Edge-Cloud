import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional
import threading

import numpy as np
import requests
from flask import Flask, request, jsonify, Response

import torch
import torch.nn.functional as F
from torchvision import transforms
from codecarbon import EmissionsTracker

from mvit_backbone import mobilevit_xxs
from mvit_policy_model import MobileViTWithPolicy
from criterions import score_entropy, score_margin, score_maxprob
from payloads import (
    decode_u8_image_bytes,
    u8_to_pil_rgb,
    tensor_to_payload,
    payload_to_tensor,
)

import msgpack

SESSION = requests.Session()
app = Flask(__name__)

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE_NAME = "edge_requests.jsonl"

ENERGY_LOG_FILE_NAME = "edge_energy.jsonl"
_TRACKERS_LOCK = threading.Lock()
_ACTIVE_TRACKERS: Dict[str, EmissionsTracker] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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

    if "application/msgpack" in ct or "application/x-msgpack" in ct:
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
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_model(
    device: torch.device, backbone_ckpt: str, ee_ckpt: str, num_classes: int = 10
):
    base = mobilevit_xxs()
    base.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))

    policy_model = MobileViTWithPolicy(
        base_model=base,
        exit_points=("mvit_0", "mvit_1"),
        num_classes=num_classes,
    )

    policy_model.load_state_dict(torch.load(ee_ckpt, map_location="cpu"), strict=False)
    policy_model = policy_model.to(device).eval()

    return policy_model


def offload_to_cloud(sample_id: Any, from_exit: str, feat_tensor: torch.Tensor,
                     cloud_url: str, timeout_s: float = 10.0
                     ) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[str], Optional[Dict[str, Any]]]:
    """
    Sends feature tensor to cloud /continue via msgpack, returns:
      - logits tensor (CPU) or None
      - edge_cloud_rtt_ms
      - cloud_error string or None
      - cloud_meta dict (cloud_compute_ms, total_cloud_ms, from_exit) or None
    """
    url = cloud_url.rstrip("/") + "/continue"

    req = {
        "sample_id": sample_id,
        "from_exit": from_exit,
        "features": tensor_to_payload(feat_tensor),
    }

    body = msgpack.packb(req, use_bin_type=True)
    headers = {
        "Content-Type": "application/msgpack",
        "Accept": "application/msgpack",
    }

    t0 = time.perf_counter()
    try:
        r = SESSION.post(url, data=body, headers=headers, timeout=timeout_s)
        t1 = time.perf_counter()
        rtt_ms = (t1 - t0) * 1000.0

        if r.status_code != 200:
            return None, rtt_ms, f"cloud_http_{r.status_code}: {r.text[:200]}", None

        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/msgpack" in ct:
            resp = msgpack.unpackb(r.content, raw=False)
        else:
            resp = r.json()

        if "logits" not in resp:
            return None, rtt_ms, "cloud_response_missing_logits", resp

        logits = payload_to_tensor(resp["logits"], device="cpu")
        cloud_meta = {
            "cloud_compute_ms": resp.get("cloud_compute_ms"),
            "total_cloud_ms": resp.get("total_cloud_ms"),
            "from_exit": resp.get("from_exit"),
        }
        return logits, rtt_ms, None, cloud_meta

    except Exception as e:
        t1 = time.perf_counter()
        return None, (t1 - t0) * 1000.0, str(e), None


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "edge",
            "time_utc": now_iso(),
            "cloud_url": app.config.get("CLOUD_URL"),
            "device": str(app.config.get("DEVICE")),
        }
    )


@app.post("/infer")
def infer():
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

    # --- Policy params ---
    policy = payload.get("policy", {}) or {}
    criterion = policy.get("criterion", "maxprob")
    thr0 = float(policy.get("thr0", 0.90))
    thr1 = float(policy.get("thr1", 0.95))

    allow_offload = True
    offload_from = payload.get("offload_from", "mvit_1")
    debug_final = bool(payload.get("debug_final", False))

    # --- Metrics placeholders ---
    edge_infer_ms = None
    edge_cloud_rtt_ms = None
    cloud_error = None
    cloud_compute_ms = None
    cloud_total_ms = None

    decision = None
    s0 = None
    s1 = None
    pred = None

    # =========================
    # 1) EDGE inference timing: decision + feature extraction only
    # =========================
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_edge0 = time.perf_counter()

    with torch.inference_mode():
        result = model(
            x,
            criterion=criterion,
            thr0=thr0,
            thr1=thr1,
            allow_offload=allow_offload,
            offload_from=offload_from,
            debug_compute_final=debug_final,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_edge1 = time.perf_counter()
    edge_infer_ms = (t_edge1 - t_edge0) * 1000.0

    decision = result.get("decision")
    s0 = result.get("score_exit0")
    s1 = result.get("score_exit1")

    if decision not in ("exit0", "exit1", "final", "offload"):
        return jsonify({
            "error": "invalid_decision",
            "message": f"Model returned invalid decision: {decision}",
            "result_keys": sorted(list(result.keys())),
        }), 500

    # local decisions
    if decision in ("exit0", "exit1", "final"):
        logits = result.get("logits", None)
        if logits is None:
            return jsonify({
                "error": "missing_logits",
                "message": f"Decision={decision} but result['logits'] is missing",
                "result_keys": sorted(list(result.keys())),
            }), 500
    else:
        logits = None  # offload path

    # =========================
    # 2) OFFLOAD timing: edge->cloud RTT separate
    # =========================
    if decision == "offload":
        feat = result.get("feature", None)
        if feat is None:
            return jsonify({
                "error": "missing_feature",
                "message": "Decision=offload but result['feature'] is missing",
                "result_keys": sorted(list(result.keys())),
            }), 500

        cloud_url = app.config.get("CLOUD_URL", None)
        timeout_s = app.config.get("CLOUD_TIMEOUT_S", 10.0)

        if not cloud_url:
            return jsonify({
                "error": "cloud_url_not_configured",
                "message": "CLOUD_URL is not configured."
            }), 400

        logits, edge_cloud_rtt_ms, cloud_error, cloud_meta = offload_to_cloud(
            sample_id=sample_id,
            from_exit=offload_from,
            feat_tensor=feat,
            cloud_url=cloud_url,
            timeout_s=timeout_s
        )

        if cloud_meta:
            cloud_compute_ms = cloud_meta.get("cloud_compute_ms")
            cloud_total_ms = cloud_meta.get("total_cloud_ms")

        if logits is None:
            return jsonify({
                "error": "offload_failed",
                "cloud_error": cloud_error,
                "edge_cloud_rtt_ms": edge_cloud_rtt_ms,
                "cloud_compute_ms": cloud_compute_ms,
                "cloud_total_ms": cloud_total_ms,
            }), 502

    # =========================
    # 3) Prediction + totals
    # =========================
    pred = int(logits.argmax(dim=1).item())

    t_total1 = time.perf_counter()
    total_edge_ms = (t_total1 - t_total0) * 1000.0

    resp = {
        "sample_id": sample_id,

        "decision": decision,
        "offload": (decision == "offload"),
        "pred": pred,

        # policy + scores
        "criterion": criterion,
        "thr0": thr0,
        "thr1": thr1,
        "score_exit0": s0,
        "score_exit1": s1,

        # timing decomposition
        "edge_infer_ms": edge_infer_ms,
        "edge_cloud_rtt_ms": edge_cloud_rtt_ms,
        "cloud_compute_ms": cloud_compute_ms,
        "cloud_total_ms": cloud_total_ms,
        "total_edge_ms": total_edge_ms,

        "cloud_error": cloud_error,
        "received_content_type": used_ct,
        "time_utc": now_iso(),
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

        "edge_infer_ms": edge_infer_ms,
        "edge_cloud_rtt_ms": edge_cloud_rtt_ms,
        "cloud_compute_ms": cloud_compute_ms,
        "cloud_total_ms": cloud_total_ms,
        "total_edge_ms": total_edge_ms,
        "cloud_error": cloud_error,
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
    ap.add_argument("--port", type=int, default=5001)

    ap.add_argument(
        "--cloud_url",
        type=str,
        required=False,
        default="http://147.102.131.35:5002",
        help="Cloud base URL (unused for now; offload is placeholder)",
    )
    ap.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)

    ap.add_argument("--backbone_ckpt", type=str, required=True)
    ap.add_argument("--ee_ckpt", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=10)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        device, args.backbone_ckpt, args.ee_ckpt, num_classes=args.num_classes
    )
    infer_tfm = build_infer_transform()

    app.config["CLOUD_URL"] = args.cloud_url
    app.config["LOG_DIR"] = args.log_dir
    app.config["DEVICE"] = device
    app.config["MODEL"] = model
    app.config["INFER_TFM"] = infer_tfm
    app.config["SERVICE_NAME"] = "edge"

    print(f"[edge] starting on http://{args.host}:{args.port}")
    print(f"[edge] device: {device}")
    print(f"[edge] backbone_ckpt: {args.backbone_ckpt}")
    print(f"[edge] ee_ckpt: {args.ee_ckpt}")
    print("[edge] /infer expects msgpack payload with raw CIFAR image bytes (image_u8)")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()