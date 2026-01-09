import os
import json
import csv
import time
import argparse
from collections import defaultdict

import requests


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_summary_csv(path: str, summary: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)


def safe_get_json(url: str, timeout_s: float) -> dict:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def run_ping(url: str, n: int, timeout_s: float) -> dict:
    times_ms = []
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            r = requests.get(url, timeout=timeout_s)
            r.raise_for_status()
            _ = r.json()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        except Exception:
            errors += 1

    if times_ms:
        avg = sum(times_ms) / len(times_ms)
        p50 = sorted(times_ms)[len(times_ms) // 2]
        p95 = sorted(times_ms)[max(0, int(0.95 * len(times_ms)) - 1)]
    else:
        avg = p50 = p95 = None

    return {
        "count": n,
        "ok": len(times_ms),
        "errors": errors,
        "avg_ms": avg,
        "p50_ms": p50,
        "p95_ms": p95,
    }


def infer(edge_infer_url: str, payload: dict, timeout_s: float, session: requests.Session) -> dict:
    """
    Returns dict with:
      - client_rtt_ms
      - edge_response_json (or error info)
    """
    t0 = time.perf_counter()
    try:
        r = session.post(edge_infer_url, json=payload, timeout=timeout_s)
        t1 = time.perf_counter()
        client_rtt_ms = (t1 - t0) * 1000.0

        # Even if non-200, capture body for debugging
        try:
            body = r.json()
        except Exception:
            body = {"raw_text": r.text[:500]}

        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "client_rtt_ms": client_rtt_ms,
            "edge_response": body,
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {
            "ok": False,
            "status_code": None,
            "client_rtt_ms": (t1 - t0) * 1000.0,
            "edge_response": {"error": str(e)},
        }


def main():
    ap = argparse.ArgumentParser(description="Client driver for Edge-Offload prototype (placeholder inference).")
    ap.add_argument("--edge_url", required=True, help="e.g. http://EDGE_IP:5001")
    ap.add_argument("--cloud_url", default=None, help="optional: e.g. http://CLOUD_IP:5002 (for ping/health)")
    ap.add_argument("--timeout_s", type=float, default=10.0)

    ap.add_argument("--n", type=int, default=10, help="number of /infer requests to send")
    ap.add_argument("--offload_rate", type=float, default=0.5, help="fraction of requests forced to offload")
    ap.add_argument("--from_exit", type=str, default="mvit_1", choices=["mvit_0", "mvit_1"])

    ap.add_argument("--log_dir", type=str, default=os.path.join("..", "logs"))
    ap.add_argument("--label", type=int, default=0, help="placeholder label to send (optional metric)")
    ap.add_argument("--sleep_ms", type=float, default=0.0, help="sleep between requests")

    args = ap.parse_args()

    edge_base = args.edge_url.rstrip("/")
    edge_health = edge_base + "/health"
    edge_infer = edge_base + "/infer"

    cloud_base = args.cloud_url.rstrip("/") if args.cloud_url else None
    cloud_health = (cloud_base + "/health") if cloud_base else None
    cloud_ping = (cloud_base + "/ping") if cloud_base else None

    log_dir = args.log_dir
    ensure_dir(log_dir)
    jsonl_path = os.path.join(log_dir, "client_requests.jsonl")
    summary_csv_path = os.path.join(log_dir, "client_summary.csv")

    session = requests.Session()

    # 1) Health checks
    print(f"[client] edge health: {edge_health}")
    edge_h = safe_get_json(edge_health, args.timeout_s)
    print("[client] edge health ok:", edge_h)

    if cloud_health:
        print(f"[client] cloud health: {cloud_health}")
        cloud_h = safe_get_json(cloud_health, args.timeout_s)
        print("[client] cloud health ok:", cloud_h)

    # 2) Ping baseline
    ping_stats = None
    if cloud_ping:
        print(f"[client] cloud ping baseline: {cloud_ping}")
        ping_stats = run_ping(cloud_ping, n=min(10, args.n), timeout_s=args.timeout_s)
        print("[client] cloud ping stats:", ping_stats)

    # 3) Main loop
    counts = defaultdict(int)
    rtts = []
    edge_compute_ms_list = []
    edge_cloud_rtt_ms_list = []
    total_edge_ms_list = []
    correct_count = 0
    correct_known = 0

    for i in range(args.n):
        force_offload = (i < int(args.n * args.offload_rate))

        payload = {
            "sample_id": i,
            "label": args.label,
            "force_offload": force_offload,
            "from_exit": args.from_exit,
            # Placeholder for future: policy dict, image tensor, thresholds, etc.
            "policy": {"action": "offload"} if force_offload else {"action": "local"},
        }

        result = infer(edge_infer, payload, timeout_s=args.timeout_s, session=session)
        rtts.append(result["client_rtt_ms"])

        edge_resp = result.get("edge_response", {})
        decision = edge_resp.get("decision", "unknown")
        counts[decision] += 1
        counts["ok"] += int(result["ok"])

        # pull numeric metrics if present
        if "edge_compute_ms" in edge_resp and edge_resp["edge_compute_ms"] is not None:
            edge_compute_ms_list.append(float(edge_resp["edge_compute_ms"]))
        if "edge_cloud_rtt_ms" in edge_resp and edge_resp["edge_cloud_rtt_ms"] is not None:
            edge_cloud_rtt_ms_list.append(float(edge_resp["edge_cloud_rtt_ms"]))
        if "total_edge_ms" in edge_resp and edge_resp["total_edge_ms"] is not None:
            total_edge_ms_list.append(float(edge_resp["total_edge_ms"]))

        corr = edge_resp.get("correct", None)
        if corr is not None:
            correct_known += 1
            correct_count += int(bool(corr))

        rec = {
            "i": i,
            "time_unix": time.time(),
            "payload": payload,
            "client_rtt_ms": result["client_rtt_ms"],
            "ok": result["ok"],
            "status_code": result["status_code"],
            "edge_response": edge_resp,
        }
        write_jsonl(jsonl_path, rec)

        print(f"[client] {i+1}/{args.n} decision={decision} rtt_ms={result['client_rtt_ms']:.2f}")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    # 4) Aggregate summary
    def avg(xs):
        return (sum(xs) / len(xs)) if xs else None

    rtt_avg = avg(rtts)
    rtt_p50 = sorted(rtts)[len(rtts) // 2] if rtts else None
    rtt_p95 = sorted(rtts)[max(0, int(0.95 * len(rtts)) - 1)] if rtts else None

    summary = {
        "time_unix": time.time(),
        "edge_url": edge_base,
        "cloud_url": cloud_base,

        "n_requests": args.n,
        "offload_rate_requested": args.offload_rate,
        "from_exit": args.from_exit,

        "ok_requests": counts["ok"],
        "decision_exit_local": counts.get("exit_local", 0),
        "decision_offload": counts.get("offload", 0),

        "client_rtt_avg_ms": rtt_avg,
        "client_rtt_p50_ms": rtt_p50,
        "client_rtt_p95_ms": rtt_p95,

        "edge_compute_avg_ms": avg(edge_compute_ms_list),
        "edge_cloud_rtt_avg_ms": avg(edge_cloud_rtt_ms_list),
        "total_edge_avg_ms": avg(total_edge_ms_list),

        "accuracy_known": (correct_count / correct_known) if correct_known > 0 else None,
    }

    # include ping stats if available
    if ping_stats:
        summary.update({
            "cloud_ping_avg_ms": ping_stats["avg_ms"],
            "cloud_ping_p50_ms": ping_stats["p50_ms"],
            "cloud_ping_p95_ms": ping_stats["p95_ms"],
            "cloud_ping_errors": ping_stats["errors"],
        })

    write_summary_csv(summary_csv_path, summary)

    print("\n[client] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n[client] wrote per-request logs: {os.path.abspath(jsonl_path)}")
    print(f"[client] wrote summary csv:      {os.path.abspath(summary_csv_path)}")


if __name__ == "__main__":
    main()
