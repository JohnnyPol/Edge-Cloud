import pandas as pd
import torch
from collections import defaultdict

from mvit_eval_policy import eval_policy
from mvit_backbone import data_loader, mobilevit_xxs
from mvit_ee_paper import MobileViTMultiExitLogits

def sweep_thresholds(model, dataloader, device, criterion, thr_list, same_thr_for_both=True):
    rows = []
    for thr in thr_list:
        thr0 = thr1 = thr if same_thr_for_both else thr  # extend later if you want separate lists

        acc, avg_ms, exits = eval_policy(
            model=model,
            dataloader=dataloader,
            device=device,
            criterion=criterion,
            thr0=thr0,
            thr1=thr1
        )

        # make sure missing exits appear as 0
        exit0 = exits.get(0, 0)
        exit1 = exits.get(1, 0)
        final = exits.get(2, 0)
        total = exit0 + exit1 + final

        rows.append({
            "criterion": criterion,
            "thr0": thr0,
            "thr1": thr1,
            "acc": acc,
            "avg_latency_ms": avg_ms,
            "exit0": exit0,
            "exit1": exit1,
            "final": final,
            "exit0_rate": exit0 / total if total else 0,
            "exit1_rate": exit1 / total if total else 0,
            "final_rate": final / total if total else 0,
        })

    return pd.DataFrame(rows)


# Prepare model and data
test_loader = data_loader(data_dir='./data',
                                  batch_size=1,
                                  test=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base = mobilevit_xxs()

# Load backbone weights first (optional if phase2 checkpoint contains full state)
base.load_state_dict(torch.load("../data/mobileViT_xxs_10.pth"))

multi = MobileViTMultiExitLogits(base_model=base, exit_points=('mvit_0','mvit_1'), num_classes=10)

# IMPORTANT: load your early-exit checkpoint into this wrapper.
# This will work ONLY if the state dict keys match.
multi.load_state_dict(torch.load("../data/mobilevit_xxs_phase2_10.pth"), strict=False)

multi = multi.to(device).eval()


thr_list_maxprob = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]
thr_list_margin = [0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.0]
thr_list_entropy = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]


df_maxprob = sweep_thresholds(multi, test_loader, device, "maxprob", thr_list_maxprob)
df_margin  = sweep_thresholds(multi, test_loader, device, "margin",  thr_list_margin)
df_entropy = sweep_thresholds(multi, test_loader, device, "entropy", thr_list_entropy)

df_all = pd.concat([df_maxprob, df_margin, df_entropy], ignore_index=True)

df_all.to_csv("../logs/sweep_results.csv", index=False)
print("Saved ../logs/sweep_results.csv")