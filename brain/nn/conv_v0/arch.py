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
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x
