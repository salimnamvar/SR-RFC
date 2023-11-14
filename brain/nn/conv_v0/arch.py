""" Conv v0 - Architectures

"""

# region Imported Dependencies
import torch
from torch import nn, Tensor


# endregion Imported Dependencies


class Arch(nn.Module):
    def __init__(self) -> None:
        super(Arch, self).__init__()

        # Define convolutional layers
        self.layers = nn.Sequential(
            nn.Conv2d(1, 484, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(484),

            nn.Conv2d(484, 242, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(242),

            nn.Conv2d(242, 120, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(120),

            nn.Conv2d(120, 60, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(60),

            nn.Conv2d(60, 30, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(30),

            nn.Conv2d(30, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x
