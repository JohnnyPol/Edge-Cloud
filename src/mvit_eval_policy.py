import torch
from collections import defaultdict

from mvit_backbone import data_loader, mobilevit_xxs
from mvit_ee_paper import MobileViTMultiExitLogits
from criterions import score_maxprob, score_entropy, score_margin

# Prepare model and data
test_loader = data_loader(data_dir='./data',
                                  batch_size=1,
                                  test=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base = mobilevit_xxs()

# Load backbone weights first (optional if phase2 checkpoint contains full state)
# Update this path to match your actual file location
base.load_state_dict(torch.load("./data/mobileViT_xxs_10.pth"))

multi = MobileViTMultiExitLogits(base_model=base, exit_points=('mvit_0','mvit_1'), num_classes=10)

# IMPORTANT: load your early-exit checkpoint into this wrapper.
# This will work ONLY if the state dict keys match.
multi.load_state_dict(torch.load("/content/mobilevit_xxs_phase2_10.pth"), strict=False)

multi = multi.to(device).eval()

images, labels = next(iter(test_loader))
images = images.to(device)

out = multi(images)

print(out.keys())
print(out["logits_mvit_0"].shape, out["logits_mvit_1"].shape, out["logits_final"].shape)


def choose_exit(outputs: dict,
                criterion: str,
                thr0: float,
                thr1: float,
                use_final_if_unsure: bool = True):
    """
    Returns: (exit_id, chosen_logits)
      exit_id: 0, 1, or 2 (2 means final)
    """
    z0 = outputs["logits_mvit_0"]
    z1 = outputs["logits_mvit_1"]
    zf = outputs["logits_final"]

    if criterion == "maxprob":
        s0 = score_maxprob(z0)
        s1 = score_maxprob(z1)
        # higher is better
        pass0 = s0 >= thr0
        pass1 = s1 >= thr1

    elif criterion == "margin":
        s0 = score_margin(z0)
        s1 = score_margin(z1)
        # higher is better
        pass0 = s0 >= thr0
        pass1 = s1 >= thr1

    elif criterion == "entropy":
        s0 = score_entropy(z0)
        s1 = score_entropy(z1)
        # lower is better for entropy (less uncertainty)
        pass0 = s0 <= thr0
        pass1 = s1 <= thr1

    else:
        raise ValueError("criterion must be one of: maxprob, margin, entropy")

    # NOTE: you are running batch_size=1, so we can use item()
    if pass0.item():
        return 0, z0
    if pass1.item():
        return 1, z1

    # not confident in early exits
    if use_final_if_unsure:
        return 2, zf
    else:
        # later we’ll replace this with “OFFLOAD”
        return 2, zf


import time
from collections import defaultdict

@torch.no_grad()
def eval_policy(model, dataloader, device, criterion, thr0, thr1):
    model.eval().to(device)

    correct = 0
    total = 0
    exit_counts = defaultdict(int)
    times = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        t0 = time.perf_counter()
        outputs = model(images)
        exit_id, logits = choose_exit(outputs, criterion=criterion, thr0=thr0, thr1=thr1)
        pred = logits.argmax(dim=1)
        t1 = time.perf_counter()

        total += labels.size(0)  # should be 1
        correct += (pred == labels).sum().item()
        exit_counts[int(exit_id)] += labels.size(0)
        times.append(t1 - t0)

    acc = correct / total
    avg_latency_ms = (sum(times) / len(times)) * 1000.0
    return acc, avg_latency_ms, dict(exit_counts)




acc, avg_ms, exits = eval_policy(
    model=multi,
    dataloader=test_loader,
    device=device,
    criterion="maxprob",
    thr0=0.85,
    thr1=0.85
)
'''
print("acc:", acc, "avg_ms:", avg_ms, "exits:", exits)
acc, avg_ms, exits = eval_policy(multi, test_loader, device, "margin", thr0=0.20, thr1=0.20)
print("acc:", acc, "avg_ms:", avg_ms, "exits:", exits)
acc, avg_ms, exits = eval_policy(multi, test_loader, device, "entropy", thr0=0.5, thr1=0.5)
print("acc:", acc, "avg_ms:", avg_ms, "exits:", exits)
'''