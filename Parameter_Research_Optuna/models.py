"""Neural network models for ENSO prediction."""

import torch
import torch.nn as nn


class HamCNN(nn.Module):
    """Convolutional neural network for multi-lead ENSO prediction."""

    def __init__(self, input_channels, leading_time, M_1=50, M_2=50, M_3=50, N_Num=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, M_1, kernel_size=(4, 8), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(M_1, M_2, kernel_size=(2, 4), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(M_2, M_3, kernel_size=(2, 4), stride=(1, 1), padding="same"),
            nn.Tanh(),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 36 * M_3, N_Num),
            nn.Linear(N_Num, leading_time),
        )

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.dense(x)


def count_trainable_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    x = torch.randn(2, 6, 49, 144)
    model = HamCNN(input_channels=6, leading_time=18, M_1=50, M_2=50, M_3=50, N_Num=50)
    print("Input :", x.shape)
    print("Output:", model(x).shape)
    print("Params:", count_trainable_parameters(model))
