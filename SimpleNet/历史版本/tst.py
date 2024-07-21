import torch
import torch.nn as nn

# 定义一个简单的卷积网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=3)

    def forward(self, x):
        return self.conv(x)

# 实例化网络
net = ConvNet()

# 定义钩子函数
def print_activation(module, input, output):
    print('钩子函数运行。 ===============')

# 注册钩子
net.conv.register_forward_hook(print_activation)

# 创建一个随机输入
input = torch.randn(1, 1, 28, 28)

# 前向传播
output = net(input)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)