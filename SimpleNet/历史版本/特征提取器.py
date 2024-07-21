import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2

# 定义一个简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self, resnet):
        super(SimpleCNN, self).__init__()
        self.resnet = resnet

        self.embeding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        self.forward_modules = torch.nn.ModuleDict({})
        self.forward_modules["layer1"] = self.resnet.layer1
        self.forward_modules["layer2"] = self.resnet.layer2
        self.forward_modules["layer3"] = self.resnet.layer3

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(in_features=2048, out_features=1000, bias=True)
        )
        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 10)  # 假设输出10维特征

    def forward(self, x):
        x = self.embeding(x)
        x = self.forward_modules["layer1"](x)
        x = self.forward_modules["layer2"](x)
        x = self.forward_modules["layer3"](x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 添加高斯噪声（不参与反向传播）
def add_gaussian_noise(images, mean=0.0, std=0.1):
    noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    return noisy_images


# KL 散度函数
def kl_divergence(p, q):
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    kl_div = F.kl_div(q.log(), p, reduction='batchmean')
    return kl_div


if __name__ == '__main__':
    resnet = wide_resnet50_2(pretrained=True, progress=True)
    model = SimpleCNN(resnet)

    # 示例数据
    T = torch.randn(18, 3, 64, 64)  # 正样本

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(10):  # 假设训练10个epoch
        model.train()

        # 添加随机噪声生成负样本（每次都不同）
        T_noisy = add_gaussian_noise(T)

        # 前向传播
        T_features = model(T)
        T_noisy_features = model(T_noisy)

        # 计算 KL 散度损失
        loss = kl_divergence(T_features, T_noisy_features)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/10], Loss: {loss.item()}')
