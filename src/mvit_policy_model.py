import torch
import torch.nn as nn
import torch.nn.functional as F

from mvit_ee import EarlyExitHead


def score_logits(logits: torch.Tensor, criterion: str) -> torch.Tensor:
    """
    Returns a score per sample (shape: [B]).
    For maxprob/margin: higher is better (more confident).
    For entropy: lower is better (more confident).
    """
    if criterion == "maxprob":
        probs = F.softmax(logits, dim=1)
        return probs.max(dim=1).values

    if criterion == "margin":
        probs = F.softmax(logits, dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    if criterion == "entropy":
        probs = F.softmax(logits, dim=1)
        eps = 1e-12
        return -(probs * (probs + eps).log()).sum(dim=1)

    raise ValueError("criterion must be one of: maxprob, margin, entropy")


def pass_threshold(score: torch.Tensor, criterion: str, thr: float) -> torch.Tensor:
    """
    Returns boolean tensor (shape [B]) indicating if we should exit.
    """
    if criterion in ("maxprob", "margin"):
        return score >= thr
    if criterion == "entropy":
        return score <= thr
    raise ValueError("criterion must be one of: maxprob, margin, entropy")


class MobileViTWithPolicy(nn.Module):
    """
    Official-style early-exit forward with flexible criterion per request.

    Forward returns a dict with:
      - decision: "exit0" | "exit1" | "offload" | "final"
      - exit_id: 0 | 1 | 2 (2 means final/offload-decision point)
      - logits: logits tensor when decision produces logits
      - feature: feature tensor when decision is offload (what to send to cloud)
      - scores: (s0, s1) floats for logging/debug
    """

    def __init__(
        self,
        base_model: nn.Module,
        exit_points=("mvit_0", "mvit_1"),
        num_classes: int = 10,
    ):
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes

        self.exit_channels = {
            "mvit_0": base_model.mvit[0].conv4[0].out_channels,
            "mvit_1": base_model.mvit[1].conv4[0].out_channels,
        }

        self.exits = nn.ModuleDict({
            name: EarlyExitHead(self.exit_channels[name], num_classes)
            for name in exit_points
        })

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        criterion: str = "maxprob",
        thr0: float = 0.90,
        thr1: float = 0.95,
        allow_offload: bool = True,
        offload_from: str = "mvit_1",
        debug_compute_final: bool = False,
    ):
        """
        allow_offload:
          - True: if not confident at exit1, return "offload" and a feature tensor.
          - False: if not confident, compute final head and return "final".

        offload_from:
          - "mvit_0" or "mvit_1" decides which feature you offload when unsure.

        debug_compute_final:
          - if True, always also compute final logits and include them in result["logits_final"].
            (Useful for debugging, not for latency experiments.)
        """

        # ---- Backbone up to mvit_0 ----
        x = self.base.conv1(x)
        x = self.base.mv2[0](x)
        x = self.base.mv2[1](x)
        x = self.base.mv2[2](x)
        x = self.base.mv2[3](x)
        x = self.base.mv2[4](x)

        x = self.base.mvit[0](x)
        feat0 = x

        # Exit0
        logits0 = self.exits["mvit_0"](feat0)
        s0 = score_logits(logits0, criterion)
        if pass_threshold(s0, criterion, thr0).item():
            out = {
                "decision": "exit0",
                "exit_id": 0,
                "logits": logits0,
                "feature": None,
                "score_exit0": float(s0.item()),
                "score_exit1": None,
            }
            if debug_compute_final:
                out["logits_final"] = self._compute_final_from_after_mvit0(feat0)
            return out

        # ---- Continue to mvit_1 ----
        x = self.base.mv2[5](feat0)
        x = self.base.mvit[1](x)
        feat1 = x

        # Exit1
        logits1 = self.exits["mvit_1"](feat1)
        s1 = score_logits(logits1, criterion)
        if pass_threshold(s1, criterion, thr1).item():
            out = {
                "decision": "exit1",
                "exit_id": 1,
                "logits": logits1,
                "feature": None,
                "score_exit0": float(s0.item()),
                "score_exit1": float(s1.item()),
            }
            if debug_compute_final:
                out["logits_final"] = self._compute_final_from_after_mvit1(feat1)
            return out

        # ---- Not confident ----
        if allow_offload:
            feature = feat0 if offload_from == "mvit_0" else feat1
            return {
                "decision": "offload",
                "exit_id": 2,
                "logits": None,
                "feature": feature,
                "score_exit0": float(s0.item()),
                "score_exit1": float(s1.item()),
                "offload_from": offload_from,
            }

        # else compute final locally
        logits_final = self._compute_final_from_after_mvit1(feat1)
        return {
            "decision": "final",
            "exit_id": 2,
            "logits": logits_final,
            "feature": None,
            "score_exit0": float(s0.item()),
            "score_exit1": float(s1.item()),
        }

    def _compute_final_from_after_mvit0(self, feat0: torch.Tensor) -> torch.Tensor:
        x = self.base.mv2[5](feat0)
        x = self.base.mvit[1](x)
        return self._compute_final_from_after_mvit1(x)

    def _compute_final_from_after_mvit1(self, feat1: torch.Tensor) -> torch.Tensor:
        x = self.base.mv2[6](feat1)
        x = self.base.mvit[2](x)
        x = self.base.conv2(x)
        x = self.base.pool(x).view(-1, x.shape[1])
        return self.base.fc(x)
