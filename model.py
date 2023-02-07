import torch
from torch import Tensor, nn

class FontClassifierModel(nn.Module):
    """
    A model to classify fonts from an image of characters.
    """
    
    def __init__(self, init_shape: tuple[int]) -> None:
        """Initiazlize the model.

        Args:
            init_shape (tuple[int]): The initial shape of the images.
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(7, 7), stride=3, padding=2),
        )
        
        demo_vec = torch.zeros((1, 3, *init_shape))
        demo_vec = self.conv_layers(demo_vec)
        
        num_features = Tensor(demo_vec).shape[2] * Tensor(demo_vec).shape[3]

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 5),
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
    