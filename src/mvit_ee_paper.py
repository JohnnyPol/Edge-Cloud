from mvit_ee import EarlyExitHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class MobileViTMultiExitLogits(nn.Module):
    """
    Forward pass returns logits for each exit and final head.
    No early-exit decisions inside the model.
    """
    def __init__(self, base_model, exit_points=('mvit_0', 'mvit_1'), num_classes=10):
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes

        # Infer exit channels from the backbone (as you did before)
        exit_channels = {
            'mvit_0': base_model.mvit[0].conv4[0].out_channels,  # 48
            'mvit_1': base_model.mvit[1].conv4[0].out_channels,  # 64
        }

        self.exits = nn.ModuleDict()
        if 'mvit_0' in exit_points:
            self.exits['mvit_0'] = EarlyExitHead(exit_channels['mvit_0'], num_classes)
        if 'mvit_1' in exit_points:
            self.exits['mvit_1'] = EarlyExitHead(exit_channels['mvit_1'], num_classes)

    def forward(self, x):
        outputs = {}

        # --- Backbone until mvit_0 ---
        x = self.base.conv1(x)
        x = self.base.mv2[0](x)
        x = self.base.mv2[1](x)
        x = self.base.mv2[2](x)
        x = self.base.mv2[3](x)
        x = self.base.mv2[4](x)

        x = self.base.mvit[0](x)
        outputs["feat_mvit_0"] = x  # optional, useful later for feature offload
        if "mvit_0" in self.exits:
            outputs["logits_mvit_0"] = self.exits["mvit_0"](x)

        # --- Backbone until mvit_1 ---
        x = self.base.mv2[5](x)
        x = self.base.mvit[1](x)
        outputs["feat_mvit_1"] = x  # optional
        if "mvit_1" in self.exits:
            outputs["logits_mvit_1"] = self.exits["mvit_1"](x)

        # --- Final head ---
        x = self.base.mv2[6](x)
        x = self.base.mvit[2](x)
        x = self.base.conv2(x)
        x = self.base.pool(x).view(x.size(0), -1)
        outputs["logits_final"] = self.base.fc(x)

        return outputs