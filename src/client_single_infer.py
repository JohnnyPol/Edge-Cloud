import argparse
import time

import numpy as np
import requests
import msgpack
from torchvision import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_url", required=True, help="e.g. http://EDGE_IP:5001")
    ap.add_argument("--data_dir", default="../data", help="where CIFAR-10 will be downloaded/cached")
    ap.add_argument("--index", type=int, default=0, help="CIFAR-10 test index to send")
    ap.add_argument("--timeout_s", type=float, default=20.0)

    ap.add_argument("--criterion", default="maxprob", choices=["maxprob", "entropy", "margin"])
    ap.add_argument("--thr0", type=float, default=0.90)
    ap.add_argument("--thr1", type=float, default=0.95)

    ap.add_argument("--accept_msgpack", action="store_true",
                    help="If set, asks server to respond with msgpack (Accept: application/msgpack).")
    args = ap.parse_args()

    edge_infer_url = args.edge_url.rstrip("/") + "/infer"

    # Load CIFAR-10 test set WITHOUT transforms (we want raw uint8 image)
    ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True)
    pil_img, label = ds[args.index]   # PIL image 32x32 RGB, label int
    img_u8 = np.array(pil_img, dtype=np.uint8)  # (32,32,3) uint8

    payload = {
        "sample_id": args.index,
        "image_u8": img_u8.tobytes(),          # raw bytes (msgpack bin)
        "shape": [32, 32, 3],
        "label": int(label),
        "policy": {
            "criterion": args.criterion,
            "thr0": float(args.thr0),
            "thr1": float(args.thr1),
        }
    }

    body = msgpack.packb(payload, use_bin_type=True)

    headers = {"Content-Type": "application/msgpack"}
    if args.accept_msgpack:
        headers["Accept"] = "application/msgpack"

    print(f"[client] sending sample idx={args.index} label={label} to {edge_infer_url}")
    t0 = time.perf_counter()
    r = requests.post(edge_infer_url, data=body, headers=headers, timeout=args.timeout_s)
    t1 = time.perf_counter()
    client_rtt_ms = (t1 - t0) * 1000.0

    print(f"[client] status={r.status_code} client_rtt_ms={client_rtt_ms:.2f}")

    # Decode response based on Content-Type
    ct = (r.headers.get("Content-Type") or "").lower()
    if "application/msgpack" in ct or "application/x-msgpack" in ct:
        resp = msgpack.unpackb(r.content, raw=False)
    else:
        resp = r.json()

    print("[client] response:")
    for k, v in resp.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
