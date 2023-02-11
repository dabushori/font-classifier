import torch
from torch import Tensor, nn
from torch.nn import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    AvgPool2d,
    Linear,
    Flatten,
    Sigmoid,
    Module,
    Softmax,
)
import numpy as np


class FontClassifierModel(Module):
    """
    A model to classify fonts from an image of characters.
    """

    def __init__(self, init_shape: tuple[int], in_channels) -> None:
        """Initiazlize the model.

        Args:
            init_shape (tuple[int]): The initial shape of the images.
        """
        super().__init__()

        self.conv_layers = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=(5, 5),
                stride=3,
                padding=2,
            ),
            ReLU(),
            Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), stride=3, padding=2
            ),
            ReLU(),
            Conv2d(
                in_channels=32, out_channels=1, kernel_size=(7, 7), stride=3, padding=2
            ),
        )

        demo_vec = torch.zeros((1, *init_shape))
        demo_vec = self.conv_layers(demo_vec)

        num_features = np.prod(demo_vec.shape)

        self.linear_layers = Sequential(
            Flatten(),
            Linear(num_features, 64),
            Sigmoid(),
            Linear(64, 32),
            Sigmoid(),
            Linear(32, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the output of the model on a given input.

        Args:
            x (torch.Tensor): The input vector.

        Returns:
            torch.Tensor: The output of the model on the input vector.
        """
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x


class Resnet32Block(Module):
    @staticmethod
    def calc_padding(shape, kernel_size, stride):
        shape = np.array(shape)
        return tuple(np.ceil(0.5 * (shape * stride - shape - stride + kernel_size - 1)).astype(int))

    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        input_shape: tuple[int],
    ):
        super().__init__()

        
        self.stride_layers = None
        if stride > 1:
            self.stride_layers = Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=Resnet32Block.calc_padding(
                        input_shape, kernel_size, stride
                    ),
                ),
                BatchNorm2d(out_channels),
            )

        self.first_layer = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=Resnet32Block.calc_padding(input_shape, kernel_size, stride),
            ),
            BatchNorm2d(out_channels),
            ReLU(),
        )

        demo_vec = torch.zeros((1, in_channels, *input_shape))
        demo_vec = self.first_layer(demo_vec)

        self.second_layer = Sequential(
            Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=Resnet32Block.calc_padding(demo_vec.shape[-2:], kernel_size, 1),
            ),
            BatchNorm2d(out_channels),
        )

        self.relu = ReLU()

    def forward(self, x):
        input_x = self.stride_layers(x) if self.stride_layers else x
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = torch.add(x, input_x)
        x = self.relu(x)
        return x


class Resnet32(Module):
    def __init__(self, input_shape: tuple[int], in_channels: int, num_classes):
        super().__init__()

        self.conv_layers = Sequential(
            # stage 1
            Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=Resnet32Block.calc_padding(input_shape, 3, 1),
            ),
            BatchNorm2d(16),
            ReLU(),
            # stage 2
            Resnet32Block(
                kernel_size=3,
                in_channels=16,
                out_channels=16,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=16,
                out_channels=16,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=16,
                out_channels=16,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=16,
                out_channels=16,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=16,
                out_channels=16,
                stride=1,
                input_shape=input_shape,
            ),
            # stage 3
            Resnet32Block(
                kernel_size=3,
                in_channels=16,
                out_channels=32,
                stride=2,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=32,
                out_channels=32,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=32,
                out_channels=32,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=32,
                out_channels=32,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=32,
                out_channels=32,
                stride=1,
                input_shape=input_shape,
            ),
            # stage 4
            Resnet32Block(
                kernel_size=3,
                in_channels=32,
                out_channels=64,
                stride=2,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=64,
                out_channels=64,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=64,
                out_channels=64,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=64,
                out_channels=64,
                stride=1,
                input_shape=input_shape,
            ),
            Resnet32Block(
                kernel_size=3,
                in_channels=64,
                out_channels=64,
                stride=1,
                input_shape=input_shape,
            ),
            AvgPool2d(kernel_size=2),
        )

        demo_vec = torch.zeros((1, 1, *input_shape))
        demo_vec = self.conv_layers(demo_vec)

        num_features = np.prod(demo_vec.shape)

        self.linear_layers = Sequential(
            Flatten(),
            Linear(in_features=num_features, out_features=num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x
