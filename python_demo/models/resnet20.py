# models/resnet20.py
# ResNet-20 for CIFAR-10. Standard architecture from He et al. 2016.
# 3 stages × 6 residual layers = 20 layers total. ~0.27M parameters.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        # 3 stages: 16, 32, 64 channels; 3 blocks each = 18 layers + 1 input + 1 fc = 20
        self.stage1 = self._make_stage(16, 16, stride=1, n=3)
        self.stage2 = self._make_stage(16, 32, stride=2, n=3)
        self.stage3 = self._make_stage(32, 64, stride=2, n=3)

        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, in_ch, out_ch, stride, n):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(n - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


if __name__ == "__main__":
    model = ResNet20()
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet-20 parameters: {total:,}")
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}  (10 classes)")
    print("ResNet-20 OK")