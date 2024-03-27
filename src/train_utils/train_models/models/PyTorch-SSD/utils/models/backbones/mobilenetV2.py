from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor
from utils.models.layers import ConvBNReLU


class _ExtraBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        intermediate_channels = out_channels // 2
        super().__init__(
            ConvBNReLU(in_channels,
                       intermediate_channels,
                       kernel_size=1,
                       relu6=True),
            ConvBNReLU(intermediate_channels,
                       intermediate_channels,
                       kernel_size=3,
                       stride=2,
                       depthwise=True,
                       relu6=True),
            ConvBNReLU(intermediate_channels,
                       out_channels,
                       kernel_size=1,
                       relu6=True),
        )


class MobileNetV2(nn.Module):
    def __init__(self, width_mult):
        super().__init__()
        trunk = mobilenet_v2(pretrained=(width_mult == 1), width_mult=width_mult)
        self.trunk = create_feature_extractor(
            trunk,
            return_nodes={
                'features.14.conv.0.2': 'C4',
                'features.18.2': 'C5',
            }
        )
        self.extra_layers = nn.ModuleList(
            [
                _ExtraBlock(trunk.last_channel, 512),
                _ExtraBlock(512, 256),
                _ExtraBlock(256, 256),
                _ExtraBlock(256, 128),
            ]
        )

    def forward(self, images):
        ftrs = self.trunk(images)
        C4 = ftrs['C4']
        C5 = ftrs['C5']
        C6 = self.extra_layers[0](C5)
        C7 = self.extra_layers[1](C6)
        C8 = self.extra_layers[2](C7)
        C9 = self.extra_layers[3](C8)
        return [C4, C5, C6, C7, C8, C9]
