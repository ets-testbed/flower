import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import List

import numpy as np


class Net(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



# ─── Model Parameter Conversion Utilities ───────────────────────────────────────────
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    current_state = net.state_dict()
    state_dict = OrderedDict()
    for key, value in zip(current_state.keys(), parameters):
        tensor = torch.as_tensor(value)
        ref_tensor = current_state[key]
        if tensor.dtype != ref_tensor.dtype:
            tensor = tensor.to(dtype=ref_tensor.dtype)
        state_dict[key] = tensor
    net.load_state_dict(state_dict, strict=True)
