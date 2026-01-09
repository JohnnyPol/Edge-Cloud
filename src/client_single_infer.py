import argparse
import time

import numpy as np
import requests
import msgpack
from torchvision import datasets


def decode_response(r: requests.Response):
    ct = (r.headers.get("Content-Type") or "").lower()
    if "application/msgpack" in ct or "application/x-msgpack" in ct:
        return msgpack.unpackb(r.content, raw=False), ct
    # try json, but fallback to text
    try:
        return r.json(), ct
    except Exception:
        return {"_raw_text": r.text[:2000]}, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_url", required=True, help="e.g. http://EDGE_IP:5001")
    ap.add_argument("--data_dir", default="../data", help="where CIFAR-10 will be downloaded/cached")
    ap.add_argument("--index", type=int, default=0, help="CIFAR-10 test index to send")
    ap.add_argument("--timeout_s", type=float, default=20.0)

    ap.add_argument("--criterion", default="maxprob", choices=["maxprob", "entropy", "margin"])
    ap.add_argument("--thr0", type=float, default=0.90)
    ap.add_argument("--thr1", type=float, default=0.95)

    ap.add_argument("--offload_from", default="mvit_1", choices=["mvit_0", "mvit_1"])
    ap.add_argument("--debug_final", action="store_true")

    ap.add_argument("--accept_msgpack", action="store_true",
                    help="Ask server to respond with msgpack (Accept: application/msgpack).")
    args = ap.parse_args()

    edge_infer_url = args.edge_url.rstrip("/") + "/infer"

    ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True)
    pil_img, true_label = ds[args.index]
    img_u8 = np.array(pil_img, dtype=np.uint8)

    payload = {
        "sample_id": int(args.index),
        "image_u8": img_u8.tobytes(),
        "shape": [32, 32, 3],
        "policy": {
            "criterion": args.criterion,
            "thr0": float(args.thr0),
            "thr1": float(args.thr1),
        },
        "offload_from": args.offload_from,
        "debug_final": bool(args.debug_final),
    }

    body = msgpack.packb(payload, use_bin_type=True)
    headers = {"Content-Type": "application/msgpack"}
    if args.accept_msgpack:
        headers["Accept"] = "application/msgpack"

    print(f"[client] sending idx={args.index} true_label={true_label} to {edge_infer_url}")
    print(f"[client] policy: {payload['policy']} offload_from={args.offload_from} debug_final={args.debug_final}")

    t0 = time.perf_counter()
    r = requests.post(edge_infer_url, data=body, headers=headers, timeout=args.timeout_s)
    t1 = time.perf_counter()
    client_rtt_ms = (t1 - t0) * 1000.0

    resp, resp_ct = decode_response(r)

    print(f"[client] status={r.status_code} client_rtt_ms={client_rtt_ms:.2f} resp_ct={resp_ct}")

    # If server errored, print what we got and stop
    if r.status_code != 200:
        print("[client] ERROR response:")
        if isinstance(resp, dict):
            for k, v in resp.items():
                print(f"  {k}: {v}")
        else:
            print(resp)
        return

    pred = resp.get("pred", None)
    correct = None
    if pred is not None:
        try:
            correct = (int(pred) == int(true_label))
        except Exception:
            correct = None

    print("[client] result summary:")
    print(f"  decision: {resp.get('decision')}")
    print(f"  pred: {pred}  true: {true_label}  correct: {correct}")
    print(f"  scores: exit0={resp.get('score_exit0')} exit1={resp.get('score_exit1')}")
    print(f"  edge_compute_ms: {resp.get('edge_compute_ms')}  total_edge_ms: {resp.get('total_edge_ms')}")
    print(f"  edge_cloud_rtt_ms: {resp.get('edge_cloud_rtt_ms')}  cloud_error: {resp.get('cloud_error')}")

    print("\n[client] full response:")
    for k, v in resp.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
