import os
import json
import csv
import time
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

from mvit_backbone import data_loader, mobilevit_xxs
from mvit_ee_paper import MobileViTMultiExitLogits
from criterions import score_maxprob, score_entropy, score_margin


def choose_exit_or_offload(outputs: dict,
                           criterion: str,
                           thr0: float,
                           thr1: float):
    """
    Returns:
      decision: "exit0" | "exit1" | "offload"
      s0, s1: criterion scores (floats)
    """
    z0 = outputs["logits_mvit_0"]
    z1 = outputs["logits_mvit_1"]

    if criterion == "maxprob":
        s0 = score_maxprob(z0)
        s1 = score_maxprob(z1)
        pass0 = s0 >= thr0
        pass1 = s1 >= thr1

    elif criterion == "margin":
        s0 = score_margin(z0)
        s1 = score_margin(z1)
        pass0 = s0 >= thr0
        pass1 = s1 >= thr1

    elif criterion == "entropy":
        s0 = score_entropy(z0)
        s1 = score_entropy(z1)
        pass0 = s0 <= thr0
        pass1 = s1 <= thr1

    else:
        raise ValueError("criterion must be one of: maxprob, margin, entropy")

    # batch_size=1 => .item() safe
    if pass0.item():
        return "exit0", float(s0.item()), float(s1.item())
    if pass1.item():
        return "exit1", float(s0.item()), float(s1.item())

    return "offload", float(s0.item()), float(s1.item())


def pick_logits_for_decision(outputs: dict, decision: str, dummy_mode: str):
    """
    dummy_mode:
      - "use_final": for OFFLOAD return final logits (debug / oracle mode)
      - "unknown":  for OFFLOAD return None (clean separation)
    """
    if decision == "exit0":
        return outputs["logits_mvit_0"]
    if decision == "exit1":
        return outputs["logits_mvit_1"]

    # offload
    if dummy_mode == "use_final":
        return outputs["logits_final"]
    if dummy_mode == "unknown":
        return None

    raise ValueError("dummy_mode must be 'use_final' or 'unknown'")


@torch.no_grad()
def eval_with_logging(model, dataloader, device,
                      criterion, thr0, thr1,
                      dummy_mode,
                      out_dir,
                      max_samples=None):
    os.makedirs(out_dir, exist_ok=True)
    per_sample_path = os.path.join(out_dir, "per_sample.jsonl")
    summary_json_path = os.path.join(out_dir, "summary.json")
    summary_csv_path = os.path.join(out_dir, "summary.csv")

    model.eval().to(device)

    # Aggregates
    total = 0
    correct = 0
    offload_count = 0
    exit_counts = defaultdict(int)
    latencies_ms = []

    # For clean mode accuracy: if dummy_mode == "unknown", offloaded have no pred
    known_total = 0
    known_correct = 0

    # Write per-sample JSONL
    with open(per_sample_path, "w", encoding="utf-8") as f_jsonl:
        for i, (images, labels) in enumerate(dataloader):
            if max_samples is not None and i >= max_samples:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Accurate timing on GPU requires sync
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(images)
            decision, s0, s1 = choose_exit_or_offload(outputs, criterion, thr0, thr1)
            chosen_logits = pick_logits_for_decision(outputs, decision, dummy_mode)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            latencies_ms.append(latency_ms)

            y_true = int(labels.item())
            total += 1

            offload = (decision == "offload")
            if offload:
                offload_count += 1

            exit_counts[decision] += 1

            # Prediction handling
            if chosen_logits is None:
                y_pred = None
                is_correct = None
            else:
                y_pred = int(chosen_logits.argmax(dim=1).item())
                is_correct = (y_pred == y_true)
                correct += int(is_correct)

                known_total += 1
                known_correct += int(is_correct)

            rec = {
                "idx": i,
                "y_true": y_true,
                "decision": decision,
                "offload": offload,
                "criterion": criterion,
                "thr0": float(thr0),
                "thr1": float(thr1),
                "score_exit0": s0,
                "score_exit1": s1,
                "latency_ms": latency_ms,
                "y_pred": y_pred,
                "correct": is_correct,
            }
            f_jsonl.write(json.dumps(rec) + "\n")

    # Summary metrics
    avg_latency_ms = sum(latencies_ms) / max(1, len(latencies_ms))
    exit0_rate = exit_counts["exit0"] / max(1, total)
    exit1_rate = exit_counts["exit1"] / max(1, total)
    offload_rate = exit_counts["offload"] / max(1, total)

    # Two accuracies are useful:
    # - overall_acc: includes offloaded if dummy_mode == use_final (oracle-ish)
    # - edge_acc_known: accuracy only on samples where edge actually produced a label
    overall_acc = correct / max(1, total)
    edge_acc_known = (known_correct / max(1, known_total)) if known_total > 0 else None

    summary = {
        "total_samples": total,
        "criterion": criterion,
        "thr0": float(thr0),
        "thr1": float(thr1),
        "dummy_mode": dummy_mode,

        "overall_accuracy": overall_acc,
        "edge_accuracy_on_known": edge_acc_known,

        "avg_latency_ms": avg_latency_ms,
        "exit_distribution": dict(exit_counts),
        "exit0_rate": exit0_rate,
        "exit1_rate": exit1_rate,
        "offload_rate": offload_rate,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional CSV (one-row)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return summary


def load_model(device, backbone_ckpt, ee_ckpt, num_classes):
    base = mobilevit_xxs()

    # Note: map_location important on Jetson
    base.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))

    multi = MobileViTMultiExitLogits(
        base_model=base,
        exit_points=("mvit_0", "mvit_1"),
        num_classes=num_classes
    )

    # strict=False: allow partial matches if your ckpt has extra keys
    multi.load_state_dict(torch.load(ee_ckpt, map_location="cpu"), strict=False)
    multi = multi.to(device).eval()
    return multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--logs_dir", type=str, default="./logs")

    ap.add_argument("--criterion", type=str, default="maxprob",
                    choices=["maxprob", "entropy", "margin"])
    ap.add_argument("--thr0", type=float, required=True)
    ap.add_argument("--thr1", type=float, required=True)

    ap.add_argument("--dummy_mode", type=str, default="use_final",
                    choices=["use_final", "unknown"])

    ap.add_argument("--backbone_ckpt", type=str, default="./data/mobileViT_xxs_10.pth")
    ap.add_argument("--ee_ckpt", type=str, default="./data/mobilevit_xxs_phase2_10.pth")

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR10 test loader (your current loader supports CIFAR10)
    test_loader = data_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test=True,
        shuffle=False
    )

    model = load_model(
        device=device,
        backbone_ckpt=args.backbone_ckpt,
        ee_ckpt=args.ee_ckpt,
        num_classes=10
    )

    # Put each run in its own subfolder for cleanliness
    run_name = f"{args.criterion}_thr0_{args.thr0}_thr1_{args.thr1}_dummy_{args.dummy_mode}"
    out_dir = os.path.join(args.logs_dir, run_name)

    summary = eval_with_logging(
        model=model,
        dataloader=test_loader,
        device=device,
        criterion=args.criterion,
        thr0=args.thr0,
        thr1=args.thr1,
        dummy_mode=args.dummy_mode,
        out_dir=out_dir,
        max_samples=args.max_samples
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
