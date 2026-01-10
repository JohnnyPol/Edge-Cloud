"""
profile_exit_scores.py

Collects score distributions for exit0 and exit1 from the EDGE server
for threshold calibration.

Key idea:
- Force NOT exiting at exit0 so we always compute exit1.
- Log score_exit0 and score_exit1 for N samples.

Usage example:
  python profile_exit_scores.py \
      --edge_url http://EDGE_IP:5001 \
      --criterion entropy \
      --num_samples 100
"""

import argparse
import json
import time
from typing import List, Dict, Any

import numpy as np
import requests
import msgpack
from torchvision import datasets


def decode_response(resp: requests.Response) -> Dict[str, Any]:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "application/msgpack" in ct or "application/x-msgpack" in ct:
        return msgpack.unpackb(resp.content, raw=False)
    return resp.json()


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_url",required=True, help="e.g. http://EDGE_IP:5001")
    ap.add_argument("--criterion", required=True,
                    choices=["maxprob", "margin", "entropy"])
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--timeout_s", type=float, default=20.0)
    ap.add_argument("--out_json", default=None,
                    help="Optional: save raw scores to JSON file")

    args = ap.parse_args()

    edge_infer_url = args.edge_url.rstrip("/") + "/infer"

    # CIFAR-10 test set
    ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)

    scores_exit0: List[float] = []
    scores_exit1: List[float] = []

    # Thresholds chosen to FORCE reaching exit1
    if args.criterion == "entropy":
        thr0 = -1.0   # entropy >= 0, so exit0 never triggers
        thr1 = 1e9    # practically triggers exit1
    else:
        thr0 = 1e9    # maxprob/margin <= 1, so exit0 never triggers
        thr1 = 0.0    # practically triggers exit1

    print(f"[profile] criterion={args.criterion}")
    print(f"[profile] forcing thr0={thr0} thr1={thr1}")
    print(f"[profile] samples={len(indices)}")

    headers = {
        "Content-Type": "application/msgpack",
        "Accept": "application/msgpack",
    }

    for sample_id in indices:
        pil_img, _ = ds[int(sample_id)]
        img_u8 = np.array(pil_img, dtype=np.uint8)

        payload = {
            "sample_id": int(sample_id),
            "image_u8": img_u8.tobytes(),
            "shape": [32, 32, 3],
            "policy": {
                "criterion": args.criterion,
                "thr0": float(thr0),
                "thr1": float(thr1),
            },
            "offload_from": "mvit_1",
            "debug_final": False,
        }

        body = msgpack.packb(payload, use_bin_type=True)

        try:
            r = requests.post(
                edge_infer_url,
                data=body,
                headers=headers,
                timeout=args.timeout_s,
            )
        except Exception as e:
            print(f"[profile] request failed for sample {sample_id}: {e}")
            continue

        if r.status_code != 200:
            print(f"[profile] non-200 for sample {sample_id}: {r.status_code}")
            continue

        resp = decode_response(r)

        s0 = resp.get("score_exit0")
        s1 = resp.get("score_exit1")

        
        if s0 is not None:
            scores_exit0.append(float(s0))
        if s1 is not None:
            scores_exit1.append(float(s1))
        
        print(f"[profile] sample {sample_id}: score_exit0={s0} score_exit1={s1}")

    print("\n=== EXIT0 SCORE DISTRIBUTION ===")
    if scores_exit0:
        for k, v in summarize(scores_exit0).items():
            print(f"{k:>7}: {v:.6f}")
    else:
        print("No exit0 scores collected.")

    print("\n=== EXIT1 SCORE DISTRIBUTION ===")
    if scores_exit1:
        for k, v in summarize(scores_exit1).items():
            print(f"{k:>7}: {v:.6f}")
    else:
        print("No exit1 scores collected.")

    if args.out_json:
        out = {
            "criterion": args.criterion,
            "thr0": thr0,
            "thr1": thr1,
            "scores_exit0": scores_exit0,
            "scores_exit1": scores_exit1,
            "summary_exit0": summarize(scores_exit0) if scores_exit0 else None,
            "summary_exit1": summarize(scores_exit1) if scores_exit1 else None,
            "time_utc": time.time(),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[profile] wrote raw scores to {args.out_json}")


if __name__ == "__main__":
    main()
