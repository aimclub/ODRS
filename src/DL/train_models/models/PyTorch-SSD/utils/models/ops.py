import torch
from torch import nn


def get_output_shapes(model, input_size):
    """
    Returns:
        : A list of [channesl, size, size]. Shape: [num_stages, [3]].
    """
    inp = torch.randn([2, 3, input_size, input_size])
    with torch.no_grad():
        out = model(inp)
    return [o.shape[1:] for o in out]


def xavier_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
