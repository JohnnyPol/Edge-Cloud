import argparse
import os
import json
import csv
import time
from collections import defaultdict

import numpy as np
from torchvision import datasets

from client_transport import EdgeClient


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_summary_csv(path: str, summary: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            w.writeheader()
        w.writerow(summary)


def avg(xs):
    return (sum(xs) / len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser(description="Batch CIFAR-10 client runner (msgpack). Computes accuracy on client.")
    ap.add_argument("--edge_url", required=True, help="e.g. http://EDGE_IP:5001")
    ap.add_argument("--data_dir", default="../data", help="where CIFAR-10 will be downloaded/cached")
    ap.add_argument("--n", type=int, default=100, help="number of test samples to send")
    ap.add_argument("--start_index", type=int, default=0, help="start index in CIFAR-10 test set")
    ap.add_argument("--timeout_s", type=float, default=20.0)
    ap.add_argument("--accept_msgpack", action="store_true", help="Ask edge to respond with msgpack.")

    # Policy controls (sent to edge)
    ap.add_argument("--criterion", default="maxprob", choices=["maxprob", "entropy", "margin"])
    ap.add_argument("--thr0", type=float, default=0.90)
    ap.add_argument("--thr1", type=float, default=0.95)

    # Logging
    ap.add_argument("--log_dir", default=os.path.join("..", "logs"))
    ap.add_argument("--run_name", default=None, help="optional name to include in log filenames")
    ap.add_argument("--sleep_ms", type=float, default=0.0, help="sleep between requests (ms)")

    args = ap.parse_args()

    run_tag = args.run_name or f"cifar10_{args.criterion}_thr0_{args.thr0}_thr1_{args.thr1}_{int(time.time())}"
    ensure_dir(args.log_dir)
    jsonl_path = os.path.join(args.log_dir, f"client_batch_{run_tag}.jsonl")
    summary_csv_path = os.path.join(args.log_dir, "client_batch_summary.csv")

    client = EdgeClient(edge_url=args.edge_url, timeout_s=args.timeout_s)

    print("[client] checking edge health...")
    h = client.health()
    print("[client] edge health:", h)

    # Load CIFAR-10 test set WITHOUT transforms (we send raw uint8)
    ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True)

    policy = {"criterion": args.criterion, "thr0": float(args.thr0), "thr1": float(args.thr1)}

    counts = defaultdict(int)
    client_rtts = []
    edge_compute_ms = []
    total_edge_ms = []
    edge_cloud_rtt_ms = []

    correct = 0
    total = 0

    print(f"[client] running n={args.n} from start_index={args.start_index} policy={policy}")
    for i in range(args.n):
        idx = args.start_index + i
        pil_img, true_label = ds[idx]
        img_u8 = np.array(pil_img, dtype=np.uint8)  # (32,32,3)
        img_bytes = img_u8.tobytes()

        resp, rtt_ms, status_code, ct = client.infer_msgpack(
            sample_id=idx,
            image_u8_bytes=img_bytes,
            shape_hw3=(32, 32, 3),
            policy=policy,
            accept_msgpack=args.accept_msgpack,
        )

        total += 1
        client_rtts.append(rtt_ms)

        decision = resp.get("decision", "unknown")
        pred = resp.get("pred", None)

        counts[f"decision_{decision}"] += 1
        counts["http_200"] += int(status_code == 200)
        counts["http_non_200"] += int(status_code != 200)

        # Edge metrics (if present)
        if resp.get("edge_compute_ms") is not None:
            edge_compute_ms.append(float(resp["edge_compute_ms"]))
        if resp.get("total_edge_ms") is not None:
            total_edge_ms.append(float(resp["total_edge_ms"]))
        if resp.get("edge_cloud_rtt_ms") is not None:
            try:
                edge_cloud_rtt_ms.append(float(resp["edge_cloud_rtt_ms"]))
            except Exception:
                pass

        is_correct = None
        if pred is not None:
            try:
                is_correct = (int(pred) == int(true_label))
                correct += int(is_correct)
            except Exception:
                is_correct = None

        rec = {
            "time_unix": time.time(),
            "idx": idx,
            "true_label": int(true_label),
            "pred": pred,
            "correct": is_correct,
            "decision": decision,
            "client_rtt_ms": rtt_ms,
            "status_code": status_code,
            "response_content_type": ct,
            "edge_response": resp,
        }
        write_jsonl(jsonl_path, rec)

        if (i + 1) % 10 == 0 or i == 0:
            acc_so_far = correct / total
            print(f"[client] {i+1}/{args.n} decision={decision} rtt_ms={rtt_ms:.1f} acc={acc_so_far:.3f}")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    accuracy = correct / total if total > 0 else None

    # Simple percentiles
    rtts_sorted = sorted(client_rtts)
    p50 = rtts_sorted[len(rtts_sorted) // 2] if rtts_sorted else None
    p95 = rtts_sorted[max(0, int(0.95 * len(rtts_sorted)) - 1)] if rtts_sorted else None

    summary = {
        "time_unix": time.time(),
        "run_tag": run_tag,
        "edge_url": args.edge_url,
        "n": args.n,
        "start_index": args.start_index,
        "criterion": args.criterion,
        "thr0": args.thr0,
        "thr1": args.thr1,

        "accuracy": accuracy,

        "client_rtt_avg_ms": avg(client_rtts),
        "client_rtt_p50_ms": p50,
        "client_rtt_p95_ms": p95,

        "edge_compute_avg_ms": avg(edge_compute_ms),
        "total_edge_avg_ms": avg(total_edge_ms),
        "edge_cloud_rtt_avg_ms": avg(edge_cloud_rtt_ms),

        # decisions
        "decision_exit0": counts.get("decision_exit0", 0),
        "decision_exit1": counts.get("decision_exit1", 0),
        "decision_offload": counts.get("decision_offload", 0),
        "decision_unknown": counts.get("decision_unknown", 0),

        "http_200": counts.get("http_200", 0),
        "http_non_200": counts.get("http_non_200", 0),
    }

    write_summary_csv(summary_csv_path, summary)

    print("\n[client] done.")
    print("[client] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n[client] per-sample log: {os.path.abspath(jsonl_path)}")
    print(f"[client] summary csv:   {os.path.abspath(summary_csv_path)}")


if __name__ == "__main__":
    main()
