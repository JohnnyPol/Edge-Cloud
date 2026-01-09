import torch.nn as nn
import torch.nn.functional as F

class EarlyExitHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.exit = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(in_channels // 2, num_classes)

    def forward(self, x):
        x = self.exit(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MobileViTWithEarlyExits(nn.Module):
    def __init__(self, base_model, exit_points=['mvit_0', 'mvit_1'], num_classes=10, exit_threshold=0.9):
        super().__init__()
        self.base = base_model
        self.exit_threshold = exit_threshold
        self.num_classes = num_classes

        self.exit_channels = {
            'mvit_0': base_model.mvit[0].conv4[0].out_channels,
            'mvit_1': base_model.mvit[1].conv4[0].out_channels
        }

        self.exits = nn.ModuleDict({
            name: EarlyExitHead(self.exit_channels[name], num_classes)
            for name in exit_points
        })

    def forward(self, x, targets=None, train_mode=True):
        losses = []

        x = self.base.conv1(x)
        x = self.base.mv2[0](x)
        x = self.base.mv2[1](x)
        x = self.base.mv2[2](x)
        x = self.base.mv2[3](x)
        x = self.base.mv2[4](x)

        x = self.base.mvit[0](x)
        if 'mvit_0' in self.exits:
            out = self.exits['mvit_0'](x)
            if train_mode:
                losses.append(F.cross_entropy(out, targets))
            else:
                conf, pred = F.softmax(out, dim=1).max(dim=1)
                if conf.mean() > self.exit_threshold:
                    return 0, pred

        x = self.base.mv2[5](x)
        x = self.base.mvit[1](x)
        if 'mvit_1' in self.exits:
            out = self.exits['mvit_1'](x)
            if train_mode:
                losses.append(F.cross_entropy(out, targets))
            else:
                conf, pred = F.softmax(out, dim=1).max(dim=1)
                if conf.mean() > self.exit_threshold:
                    return 1, pred

        x = self.base.mv2[6](x)
        x = self.base.mvit[2](x)
        x = self.base.conv2(x)
        x = self.base.pool(x).view(-1, x.shape[1])
        out = self.base.fc(x)

        if train_mode:
            losses.append(F.cross_entropy(out, targets))
            return losses
        else:
            _, pred = F.softmax(out, dim=1).max(dim=1)
            return len(self.exits), pred