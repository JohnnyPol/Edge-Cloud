import torch.nn as nn
import torch.Tensor

class MobileViTCloudContinuation(nn.Module):
    """
    Runs the remaining layers given a feature tensor from exit0 or exit1.
    Assumes feature tensor is exactly what edge sends:
      - mvit_0 feature: output of base.mvit[0](...)
      - mvit_1 feature: output of base.mvit[1](...)
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    @torch.inference_mode()
    def forward_from_exit(self, feat: torch.Tensor, from_exit: str) -> torch.Tensor:
        if from_exit == "mvit_0":
            x = self.base.mv2[5](feat)
            x = self.base.mvit[1](x)
            # then fallthrough to exit1 continuation
            x = self.base.mv2[6](x)
            x = self.base.mvit[2](x)
            x = self.base.conv2(x)
            x = self.base.pool(x).view(-1, x.shape[1])
            return self.base.fc(x)

        if from_exit == "mvit_1":
            x = self.base.mv2[6](feat)
            x = self.base.mvit[2](x)
            x = self.base.conv2(x)
            x = self.base.pool(x).view(-1, x.shape[1])
            return self.base.fc(x)

        raise ValueError(f"Unknown from_exit={from_exit}. Expected 'mvit_0' or 'mvit_1'.")
