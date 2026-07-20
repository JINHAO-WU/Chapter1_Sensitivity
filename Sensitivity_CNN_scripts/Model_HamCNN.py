"""HamCNN model used for ENSO forecast experiments."""

import torch
from torch import nn


class HamCNN(nn.Module):
    """Three-stage CNN followed by dense forecast layers.

    Omitting ``input_channels`` or ``output_steps`` retains the values from
    ``A_Parameter_set`` while avoiding a module-level circular import.
    """

    def __init__(self, M_1: int, M_2: int, M_3: int, N_Num: int, input_channels=None, output_steps=None):
        super().__init__()
        if input_channels is None or output_steps is None:
            import A_Parameter_set as par
            input_channels = par.pa if input_channels is None else input_channels
            output_steps = par.leading_time if output_steps is None else output_steps

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, M_1, kernel_size=(4, 8), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(M_1, M_2, kernel_size=(2, 4), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(M_2, M_3, kernel_size=(2, 4), padding="same"),
            nn.Tanh(),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(M_3 * 12 * 36, N_Num), # 12 36, 4 14
            nn.Linear(N_Num, output_steps),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inputs)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.dense(x)
        return x
        #return self.dense(self.conv3(self.conv2(self.conv1(inputs))))


if __name__ == "__main__":
    model = HamCNN(64, 32, 16, 256, input_channels=6, output_steps=18)
    output = model(torch.randn(2, 6, 49, 144))
    print(f"Output: {output.shape}")
