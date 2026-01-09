import torch
import torch.nn.functional as F

def score_maxprob(logits: torch.Tensor) -> torch.Tensor:
    # returns shape (B,)
    probs = F.softmax(logits, dim=1)
    return probs.max(dim=1).values

def score_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # normalized entropy is optional; here we return raw entropy (higher = more uncertain)
    probs = F.softmax(logits, dim=1)
    return -(probs * (probs + eps).log()).sum(dim=1)

def score_margin(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values  # (B,2)
    return top2[:, 0] - top2[:, 1]

