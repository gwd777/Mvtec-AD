import torch
import torch.nn as nn
from gym.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class Discriminator(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 512):
        super(Discriminator, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[1]  # 1536
        self.body = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1024, bias=True),
            # nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.tail = nn.Linear(in_features=1024, out_features=2, bias=False)

        # The features_dim is the size of the output features
        self._features_dim = 2

    def forward(self, x):
        # 这里我们假设输入形状为 [batch_size, 1296, 1536]
        batch_size, sequence_length, feature_dim = x.shape
        print('_____________x.shape before processing_______________>', x.shape)

        # 将每个时间步（sequence step）的特征通过body处理
        x = self.body(x)  # shape: [batch_size, 1296, 1024]
        print('_____________x.shape after body_______________>', x.shape)

        # 对时间步（sequence step）维度取平均
        x = torch.mean(x, dim=1)  # shape: [batch_size, 1024]
        print('_____________x.shape after mean_______________>', x.shape)

        # 通过尾部的线性层
        x = self.tail(x)  # shape: [batch_size, 2]
        print('_____________x.shape after tail_______________>', x.shape)

        return x


# 定义观测空间，假设输入数据的值范围是0到255
observation_space = Box(low=0, high=255, shape=(1296, 1536), dtype=np.float32)

# 创建判别器模型
discriminator = Discriminator(observation_space)

# 示例使用，输入张量形状为 [batch_size, 1296, 1536]
input_tensor = torch.randn(5, 1296, 1536)  # 这里batch_size为1
output = discriminator(input_tensor)
print('Output:', output)
