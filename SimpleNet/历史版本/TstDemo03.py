import torch
import torch.nn as nn
from gym.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Discriminator(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 512):
        super(Discriminator, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0] * observation_space.shape[1]
        self.body = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1024, bias=True),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.tail = nn.Linear(in_features=1024, out_features=2, bias=False)

        # The features_dim is the size of the output features
        self._features_dim = 2

    def forward(self, x):
        # Flatten the input tensor
        x = torch.flatten(x, start_dim=1)
        print('_____________x.shape after flatten_______________>', x.shape)
        x = self.body(x)
        print('_____________x.shape after body_______________>', x.shape)
        x = torch.mean(x, dim=0, keepdim=True)
        print('_____________x.shape after mean_______________>', x.shape)
        x = self.tail(x)
        print('_____________x.shape after tail_______________>', x.shape)
        return x

# Define the observation space with shape (1296, 1536)
observation_space = Box(low=-1, high=1, shape=(1296, 1536))

# Create the discriminator model
discriminator = Discriminator(observation_space)

# Example usage with input tensor of shape [1, 1296, 1536]
input_tensor = torch.randn(1, 1296, 1536)
output = discriminator(input_tensor)
print('Output:', output)
