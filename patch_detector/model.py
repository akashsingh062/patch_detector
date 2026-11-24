"""Global and local evidence from a single RGB encoder."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class PatchDetector(nn.Module):
    def __init__(self, local=True, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                     net.layer1, net.layer2, net.layer3)
        self.global_head = nn.Sequential(net.layer4, net.avgpool, nn.Flatten(), nn.Linear(512, 1))
        self.patch_head = nn.Conv2d(256, 1, 1) if local else None

    def forward(self, image):
        features = self.encoder(image)
        global_score = self.global_head(features).flatten()
        if self.patch_head is None:
            return global_score
        scores = self.patch_head(features).flatten(1)
        # At 256 px input, pool the strongest 32 of 256 overlapping feature locations.
        count = max(1, scores.size(1) // 8)
        local_score = torch.topk(scores, count, dim=1).values.mean(dim=1)
        return (global_score + local_score) / 2
