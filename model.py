# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "Xception",
    "SeparableConv2d", "XceptionBlock",
    "xception",
]


class Xception(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
    ) -> None:
        super(Xception, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(True)

        self.block1 = XceptionBlock(64, 128, 2, False, True, 2)
        self.block2 = XceptionBlock(128, 256, 2, True, True, 2)
        self.block3 = XceptionBlock(256, 728, 2, True, True, 2)

        self.block4 = XceptionBlock(728, 728, 1, True, True, 3)
        self.block5 = XceptionBlock(728, 728, 1, True, True, 3)
        self.block6 = XceptionBlock(728, 728, 1, True, True, 3)
        self.block7 = XceptionBlock(728, 728, 1, True, True, 3)

        self.block8 = XceptionBlock(728, 728, 1, True, True, 3)
        self.block9 = XceptionBlock(728, 728, 1, True, True, 3)
        self.block10 = XceptionBlock(728, 728, 1, True, True, 3)
        self.block11 = XceptionBlock(728, 728, 1, True, True, 3)

        self.block12 = XceptionBlock(728, 1024, 2, True, False, 2)

        self.conv3 = SeparableConv2d(1024, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(True)

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.global_average_pooling(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module, "stddev") else 0.1
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs: Any
    ) -> None:
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, groups=in_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.pointwise(out)

        return out


class XceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            relu_first: bool,
            grow_first: bool,
            repeat_times: int,
    ) -> None:
        super(XceptionBlock, self).__init__()
        rep = []

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride),
                                  padding=(0, 0), bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        mid_channels = in_channels
        if grow_first:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(out_channels))
            mid_channels = out_channels

        for _ in range(repeat_times - 1):
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(mid_channels))

        if not grow_first:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(out_channels))

        if not relu_first:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(False)

        if stride != 1:
            rep.append(nn.MaxPool2d((3, 3), (stride, stride), (1, 1)))

        self.rep = nn.Sequential(*rep)

    def forward(self, x: Tensor) -> Tensor:
        if self.skip is not None:
            identity = self.skip(x)
            identity = self.skipbn(identity)
        else:
            identity = x

        out = self.rep(x)
        out = torch.add(out, identity)

        return out


def xception(**kwargs: Any) -> Xception:
    model = Xception(**kwargs)

    return model
