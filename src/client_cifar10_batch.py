"""
client_cifar10_batch.py

Batch sweep client for:
"Uncertainty-Aware Early-Exit Inference with Selective Offloading in Edge–Cloud Systems"

- Sends CIFAR-10 samples to EDGE /infer via msgpack
- EDGE may early-exit locally or offload to CLOUD (offload_from = mvit_1 fixed)
- Computes accuracy on client (true labels never sent to edge)
- Aggregates per-policy metrics and writes a CSV similar in spirit to MobileViT-EE reports
- Writes per-request JSONL logs for reproducibility/debugging
- Optionally calls /energy/start and /energy/stop on both edge+cloud if those endpoints exist

Author: you + ChatGPT
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np
import requests
from torchvision import datasets


# -----------------------------
# Criterion-specific grids
# -----------------------------
def policy_grids() -> Dict[str, Dict[str, List[float]]]:
    """
    Semantics:
    - maxprob/margin: higher score means more confident -> exit if score >= thr
    - entropy: lower score means more confident -> exit if score <= thr

    Keep thr0 generally less strict than thr1.
    """
    return {
        "maxprob": {
            "thr0": [0.55, 0.65, 0.75, 0.85, 0.92],
            "thr1": [0.75, 0.85, 0.92, 0.96, 0.99],
        },
        "margin": {
            "thr0": [0.30, 0.45, 0.60, 0.75, 0.85],
            "thr1": [0.60, 0.75, 0.85, 0.92, 0.97],
        },
        "entropy": {
            "thr0": [1.5, 1.2, 1.0, 0.8, 0.6, 0.4],
            "thr1": [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
        },
    }



# -----------------------------
# Helpers
# -----------------------------
def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def decode_response(resp: requests.Response) -> Tuple[Dict[str, Any], str]:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "application/msgpack" in ct or "application/x-msgpack" in ct:
        try:
            return msgpack.unpackb(resp.content, raw=False), ct
        except Exception:
            return {"error": "client_decode_failed", "raw_len": len(resp.content)}, ct
    try:
        return resp.json(), ct
    except Exception:
        return {"error": "client_decode_failed", "_raw_text": resp.text[:2000]}, ct


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    arr = np.array(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def extract_energy_metrics(d: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (energy_kwh, co2_kg, duration_s) from an /energy/stop response dict.
    Supports a few key variants for robustness.
    """
    if not isinstance(d, dict):
        return None, None, None

    # Preferred keys (we'll implement these in your endpoints)
    energy_kwh = safe_float(d.get("energy_kwh"))
    co2_kg = safe_float(d.get("co2_kg"))
    duration_s = safe_float(d.get("duration_s"))

    return energy_kwh, co2_kg, duration_s

# -----------------------------
# Energy endpoint stubs (optional)
# -----------------------------
def energy_start(
    session: requests.Session, base_url: str, run_id: str, timeout_s: float
) -> Optional[Dict[str, Any]]:
    """
    Calls POST {base_url}/energy/start if exists.
    If endpoint is missing or returns non-2xx, returns None.
    """
    url = base_url.rstrip("/") + "/energy/start"
    payload = {"run_id": run_id}
    body = msgpack.packb(payload, use_bin_type=True)
    headers = {"Content-Type": "application/msgpack", "Accept": "application/msgpack"}
    try:
        r = session.post(url, data=body, headers=headers, timeout=timeout_s)
        if r.status_code // 100 != 2:
            return None
        data, _ = decode_response(r)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def energy_stop(
    session: requests.Session, base_url: str, run_id: str, timeout_s: float
) -> Optional[Dict[str, Any]]:
    """
    Calls POST {base_url}/energy/stop if exists.
    If endpoint is missing or returns non-2xx, returns None.
    """
    url = base_url.rstrip("/") + "/energy/stop"
    payload = {"run_id": run_id}
    body = msgpack.packb(payload, use_bin_type=True)
    headers = {"Content-Type": "application/msgpack", "Accept": "application/msgpack"}
    try:
        r = session.post(url, data=body, headers=headers, timeout=timeout_s)
        if r.status_code // 100 != 2:
            return None
        data, _ = decode_response(r)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# -----------------------------
# Core run logic
# -----------------------------
@dataclass
class SweepResult:
    criterion: str
    thr0: float
    thr1: float
    n_total: int
    n_ok: int
    n_fail: int
    accuracy: Optional[float]

    exit0_pct: float
    exit1_pct: float
    offload_pct: float
    final_pct: float

    client_rtt_avg: Optional[float]
    client_rtt_p95: Optional[float]

    total_edge_avg: Optional[float]
    total_edge_p95: Optional[float]

    edge_infer_avg: Optional[float]
    edge_infer_p95: Optional[float]

    edge_cloud_rtt_avg: Optional[float]
    edge_cloud_rtt_p95: Optional[float]

    cloud_compute_avg: Optional[float]
    cloud_compute_p95: Optional[float]

    total_cloud_avg: Optional[float]
    total_cloud_p95: Optional[float]

    # Energy totals per sweep (optional, filled if endpoints exist)
    edge_energy_kwh: Optional[float]
    edge_co2_kg: Optional[float]
    cloud_energy_kwh: Optional[float]
    cloud_co2_kg: Optional[float]


def run_sweep(
    session: requests.Session,
    edge_infer_url: str,
    ds: datasets.CIFAR10,
    indices: List[int],
    criterion: str,
    thr0: float,
    thr1: float,
    timeout_s: float,
    accept_msgpack: bool,
    jsonl_path: str,
    run_id: str,
) -> SweepResult:
    headers = {"Content-Type": "application/msgpack"}
    if accept_msgpack:
        headers["Accept"] = "application/msgpack"

    # Counters
    n_total = len(indices)
    n_ok = 0
    n_fail = 0
    correct = 0

    decision_counts = {"exit0": 0, "exit1": 0, "offload": 0, "final": 0, "other": 0}

    # Latency collections
    client_rtts: List[float] = []
    total_edges: List[float] = []
    edge_infers: List[float] = []

    edge_cloud_rtts: List[float] = []
    cloud_computes: List[float] = []
    total_clouds: List[float] = []

    for sample_id in indices:
        pil_img, true_label = ds[sample_id]
        img_u8 = np.array(pil_img, dtype=np.uint8)

        payload = {
            "sample_id": int(sample_id),
            "image_u8": img_u8.tobytes(),
            "shape": [32, 32, 3],
            "policy": {
                "criterion": criterion,
                "thr0": float(thr0),
                "thr1": float(thr1),
            },
            # fixed by design requirement
            "offload_from": "mvit_1",
            "debug_final": False,
        }

        body = msgpack.packb(payload, use_bin_type=True)

        t0 = time.perf_counter()
        try:
            r = session.post(
                edge_infer_url, data=body, headers=headers, timeout=timeout_s
            )
            t1 = time.perf_counter()
            client_rtt_ms = (t1 - t0) * 1000.0
        except Exception as e:
            t1 = time.perf_counter()
            client_rtt_ms = (t1 - t0) * 1000.0

            n_fail += 1
            append_jsonl(
                jsonl_path,
                {
                    "run_id": run_id,
                    "sample_id": int(sample_id),
                    "true_label": int(true_label),
                    "criterion": criterion,
                    "thr0": float(thr0),
                    "thr1": float(thr1),
                    "client_rtt_ms": client_rtt_ms,
                    "http_status": None,
                    "error": "client_request_failed",
                    "exception": str(e),
                    "ts_client": time.time(),
                },
            )
            continue

        resp, resp_ct = decode_response(r)
        http_status = r.status_code

        rec: Dict[str, Any] = {
            "run_id": run_id,
            "sample_id": int(sample_id),
            "true_label": int(true_label),
            "criterion": criterion,
            "thr0": float(thr0),
            "thr1": float(thr1),
            "client_rtt_ms": client_rtt_ms,
            "http_status": http_status,
            "resp_ct": resp_ct,
            "ts_client": time.time(),
        }

        if http_status != 200 or not isinstance(resp, dict):
            n_fail += 1
            rec["error"] = "edge_non_200" if http_status != 200 else "bad_response_type"
            rec["resp"] = resp
            append_jsonl(jsonl_path, rec)
            continue

        # Success path
        n_ok += 1
        client_rtts.append(client_rtt_ms)

        decision = resp.get("decision", None)
        if decision in decision_counts:
            decision_counts[decision] += 1
        else:
            decision_counts["other"] += 1

        pred = resp.get("pred", None)
        is_correct = None
        if pred is not None:
            try:
                is_correct = int(pred) == int(true_label)
                if is_correct:
                    correct += 1
            except Exception:
                is_correct = None

        # Pull server timings (robustly)
        edge_infer_ms = safe_float(resp.get("edge_infer_ms"))
        total_edge_ms = safe_float(resp.get("total_edge_ms"))
        edge_cloud_rtt_ms = safe_float(resp.get("edge_cloud_rtt_ms"))
        cloud_compute_ms = safe_float(resp.get("cloud_compute_ms"))
        total_cloud_ms = safe_float(resp.get("total_cloud_ms"))

        if edge_infer_ms is not None:
            edge_infers.append(edge_infer_ms)
        if total_edge_ms is not None:
            total_edges.append(total_edge_ms)

        # Offload-only distributions
        if edge_cloud_rtt_ms is not None:
            edge_cloud_rtts.append(edge_cloud_rtt_ms)
        if cloud_compute_ms is not None:
            cloud_computes.append(cloud_compute_ms)
        if total_cloud_ms is not None:
            total_clouds.append(total_cloud_ms)

        # Store record
        rec.update(
            {
                "decision": decision,
                "pred": pred,
                "correct": is_correct,
                # scores
                "score_exit0": resp.get("score_exit0"),
                "score_exit1": resp.get("score_exit1"),
                # timing fields from server
                "edge_infer_ms": edge_infer_ms,
                "edge_cloud_rtt_ms": edge_cloud_rtt_ms,
                "cloud_compute_ms": cloud_compute_ms,
                "total_cloud_ms": total_cloud_ms,
                "total_edge_ms": total_edge_ms,
                "cloud_error": resp.get("cloud_error"),
            }
        )
        append_jsonl(jsonl_path, rec)

    accuracy = (correct / n_ok) if n_ok > 0 else None

    exit0_pct = (decision_counts["exit0"] / n_ok) * 100.0 if n_ok else 0.0
    exit1_pct = (decision_counts["exit1"] / n_ok) * 100.0 if n_ok else 0.0
    offload_pct = (decision_counts["offload"] / n_ok) * 100.0 if n_ok else 0.0
    final_pct = (decision_counts["final"] / n_ok) * 100.0 if n_ok else 0.0

    return SweepResult(
        criterion=criterion,
        thr0=float(thr0),
        thr1=float(thr1),
        n_total=n_total,
        n_ok=n_ok,
        n_fail=n_fail,
        accuracy=accuracy,
        exit0_pct=exit0_pct,
        exit1_pct=exit1_pct,
        offload_pct=offload_pct,
        final_pct=final_pct,
        client_rtt_avg=mean(client_rtts),
        client_rtt_p95=percentile(client_rtts, 95),
        total_edge_avg=mean(total_edges),
        total_edge_p95=percentile(total_edges, 95),
        edge_infer_avg=mean(edge_infers),
        edge_infer_p95=percentile(edge_infers, 95),
        edge_cloud_rtt_avg=mean(edge_cloud_rtts),
        edge_cloud_rtt_p95=percentile(edge_cloud_rtts, 95),
        cloud_compute_avg=mean(cloud_computes),
        cloud_compute_p95=percentile(cloud_computes, 95),
        total_cloud_avg=mean(total_clouds),
        total_cloud_p95=percentile(total_clouds, 95),
        edge_energy_kwh=None,
        edge_co2_kg=None,
        cloud_energy_kwh=None,
        cloud_co2_kg=None,
    )


def write_csv(csv_path: str, rows: List[SweepResult]) -> None:
    ensure_parent_dir(csv_path)
    fieldnames = [
        "criterion",
        "thr0",
        "thr1",
        "n_total",
        "n_ok",
        "n_fail",
        "accuracy",
        "exit0_pct",
        "exit1_pct",
        "offload_pct",
        "final_pct",
        "client_rtt_avg",
        "client_rtt_p95",
        "total_edge_avg",
        "total_edge_p95",
        "edge_infer_avg",
        "edge_infer_p95",
        "edge_cloud_rtt_avg",
        "edge_cloud_rtt_p95",
        "cloud_compute_avg",
        "cloud_compute_p95",
        "total_cloud_avg",
        "total_cloud_p95",
        "edge_energy_kwh",
        "edge_co2_kg",
        "cloud_energy_kwh",
        "cloud_co2_kg",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_url", required=True, help="e.g. http://EDGE_IP:5001")
    ap.add_argument(
        "--cloud_url",
        required=True,
        help="Optional: for energy endpoints (same host as /continue).",
    )
    ap.add_argument(
        "--data_dir", default="../data", help="CIFAR-10 download/cache directory"
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of CIFAR-10 test samples per sweep",
    )
    ap.add_argument(
        "--seed", type=int, default=123, help="Seed for deterministic sampling"
    )
    ap.add_argument("--timeout_s", type=float, default=20.0)
    ap.add_argument(
        "--accept_msgpack",
        action="store_true",
        help="Request msgpack responses from edge.",
    )

    ap.add_argument("--out_csv", default="../logs/results.csv")
    ap.add_argument("--out_jsonl", default="../logs/client_requests.jsonl")

    ap.add_argument(
        "--dry_run", action="store_true", help="Print sweeps but don't execute."
    )
    args = ap.parse_args()

    edge_infer_url = args.edge_url.rstrip("/") + "/infer"
    cloud_url = args.cloud_url.rstrip("/") if args.cloud_url else None

    # Dataset
    ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True)

    # Deterministic indices (no label leak: we only use labels locally)
    if args.num_samples is None or args.num_samples <= 0:
    # Full CIFAR-10 test set
        indices = list(range(len(ds)))
    else:
        rng = np.random.default_rng(args.seed)
        all_indices = np.arange(len(ds))
        chosen = rng.choice(
            all_indices,
            size=min(args.num_samples, len(ds)),
            replace=False,
        )
        indices = [int(x) for x in chosen.tolist()]

    grids = policy_grids()

    # Build sweep list
    sweep_triplets: List[Tuple[str, float, float]] = []
    for criterion, g in grids.items():
        thr0_list = g["thr0"]
        thr1_list = g["thr1"]
        for thr0 in thr0_list:
            for thr1 in thr1_list:
                # optional: enforce thr1 stricter than thr0 for monotonicity
                # - for maxprob/margin: thr1 >= thr0
                # - for entropy: thr1 <= thr0
                if criterion in ("maxprob", "margin"):
                    if thr1 < thr0:
                        continue
                else:  # entropy
                    if thr1 > thr0:
                        continue
                sweep_triplets.append((criterion, float(thr0), float(thr1)))

    print(f"[client] edge_infer_url: {edge_infer_url}")
    if args.num_samples <= 0:
        print(f"[client] using FULL CIFAR-10 test set ({len(indices)} samples)")
    else:
        print(f"[client] num_samples per sweep: {len(indices)}  seed={args.seed}")
    print(f"[client] sweeps: {len(sweep_triplets)} total (criterion-specific grids)")
    print(f"[client] out_csv: {os.path.abspath(args.out_csv)}")
    print(f"[client] out_jsonl: {os.path.abspath(args.out_jsonl)}")

    if args.dry_run:
        for c, t0, t1 in sweep_triplets[:30]:
            print(f"  - {c} thr0={t0} thr1={t1}")
        if len(sweep_triplets) > 30:
            print("  ...")
        return

    session = requests.Session()
    results: List[SweepResult] = []

    # Run sweeps
    for i, (criterion, thr0, thr1) in enumerate(sweep_triplets, start=1):
        run_id = f"{criterion}_thr0={thr0:.4f}_thr1={thr1:.4f}_seed={args.seed}_n={len(indices)}"
        print(f"\n[client] sweep {i}/{len(sweep_triplets)}  {run_id}")

        # Energy tracking (optional; endpoints not implemented yet)
        edge_energy = None
        cloud_energy = None
        _ = energy_start(
            session, args.edge_url, run_id=run_id, timeout_s=args.timeout_s
        )
        if cloud_url:
            _ = energy_start(
                session, cloud_url, run_id=run_id, timeout_s=args.timeout_s
            )

        res = run_sweep(
            session=session,
            edge_infer_url=edge_infer_url,
            ds=ds,
            indices=indices,
            criterion=criterion,
            thr0=thr0,
            thr1=thr1,
            timeout_s=args.timeout_s,
            accept_msgpack=args.accept_msgpack,
            jsonl_path=args.out_jsonl,
            run_id=run_id,
        )

        # Stop energy tracking (optional)
        edge_stop = energy_stop(
            session, args.edge_url, run_id=run_id, timeout_s=args.timeout_s
        )
        cloud_stop = None
        if cloud_url:
            cloud_stop = energy_stop(
                session, cloud_url, run_id=run_id, timeout_s=args.timeout_s
            )

        edge_energy_kwh, edge_co2_kg, _edge_dur_s = extract_energy_metrics(edge_stop)
        cloud_energy_kwh, cloud_co2_kg, _cloud_dur_s = extract_energy_metrics(cloud_stop)

        # Attach to SweepResult
        res.edge_energy_kwh = edge_energy_kwh
        res.edge_co2_kg = edge_co2_kg
        res.cloud_energy_kwh = cloud_energy_kwh
        res.cloud_co2_kg = cloud_co2_kg

        results.append(res)

        print(
            f"[client] n_ok={res.n_ok}/{res.n_total} acc={res.accuracy if res.accuracy is not None else 'NA'}"
        )
        print(
            f"[client] exits: exit0={res.exit0_pct:.1f}% exit1={res.exit1_pct:.1f}% offload={res.offload_pct:.1f}%"
        )
        print(f"[client] client_rtt avg={res.client_rtt_avg} p95={res.client_rtt_p95}")
        print(f"[client] total_edge avg={res.total_edge_avg} p95={res.total_edge_p95}")

        # Write CSV incrementally for safety
        write_csv(args.out_csv, results)

    print("\n[client] done.")
    print(f"[client] wrote: {os.path.abspath(args.out_csv)}")
    print(f"[client] wrote: {os.path.abspath(args.out_jsonl)}")


if __name__ == "__main__":
    main()
