import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(in_features=1536, out_features=1024, bias=True),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # 最后一个线性层转换为期望输出形状
        self.tail = nn.Linear(in_features=1024, out_features=2, bias=False)

    def forward(self, x):
        x = self.body(x)
        # 对所有输入进行全局平均池化
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.tail(x)
        return x

# 实例化并测试 Discriminator
discriminator = Discriminator()
input_tensor = torch.randn(1296, 1536)  # (1296, 1536)
output = discriminator(input_tensor)
print(output.shape)  # 应该打印 torch.Size([1, 2])
