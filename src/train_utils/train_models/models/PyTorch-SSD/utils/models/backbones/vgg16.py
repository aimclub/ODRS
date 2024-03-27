import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
from utils.models.ops import xavier_init


class _ExtraBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, padding=0, stride=1):
        intermediate_channels = out_channels // 2
        super().__init__(
            nn.Conv2d(in_channels,
                      intermediate_channels,
                      kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=padding),
            nn.ReLU(inplace=True),
        )


class L2_Normalize(nn.Module):
    def __init__(self, channels, s0=20):
        super().__init__()
        self.s = nn.Parameter(torch.full([channels, 1, 1], s0, dtype=torch.float32))

    def forward(self, x):
        return self.s * F.normalize(x, p=2, dim=1)


class VGG16(nn.Module):
    def __init__(self, num_stages):
        super().__init__()
        if not os.path.exists('vgg16_features-amdegroot-88682ab5.pth'):
            import wget
            wget.download('https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth')   # noqa: E501
        trunk = vgg16(pretrained=False)
        trunk.load_state_dict(torch.load('vgg16_features-amdegroot-88682ab5.pth'))

        trunk.features[16].ceil_mode = True   # pool3
        trunk.features[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)   # pool5
        self.trunk = create_feature_extractor(
            trunk,
            return_nodes={
                'features.22': 'C3',   # conv4_3
                'features.30': 'C4',   # pool5
            }
        )
        self.l2_normalize = L2_Normalize(512)

        self.extra_layers = nn.ModuleList(
            [
                nn.Conv2d(512, 1024, kernel_size=3, dilation=6, padding=6),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=1),
                nn.ReLU(inplace=True),
                _ExtraBlock(1024, 512, padding=1, stride=2),
                _ExtraBlock(512, 256, padding=1, stride=2),
            ]
        )
        for i in range(num_stages - 4):
            self.extra_layers.append(_ExtraBlock(256, 256))

        self.extra_layers.apply(xavier_init)

    def forward(self, images):
        ftrs = self.trunk(images)
        outputs = [self.l2_normalize(ftrs['C3'])]
        x = ftrs['C4']
        for i, layer in enumerate(self.extra_layers):
            x = layer(x)
            if i > 2:
                outputs.append(x)
        return outputs
