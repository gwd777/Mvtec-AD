import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
from tqdm import tqdm
from datasets.mvtec import MVTecDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('________当前设备_________>', device)


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
        self.fc2 = nn.Linear(128, 2)  # 假设输出10维特征

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

    data_path = '/home/wzx_python_project/'
    class_name = 'big_hw_ls_1000data'
    epoch_num = 50
    best_loss = float('inf')  # 初始化为正无穷大
    best_model_state = None

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train_dataset = MVTecDataset(data_path, class_name=class_name, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    model.train()
    model.to(device)

    for epoch in range(epoch_num):
        epoch_loss = 0  # 用于累加每个epoch的总损失
        # 训练循环
        for (img, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            T_noisy = add_gaussian_noise(img)       # 添加随机噪声生成负样本（每次都不同）

            # 前向传播
            T_features = model(img)
            T_noisy_features = model(T_noisy)
            loss = kl_divergence(T_features, T_noisy_features)      # 计算 KL 散度损失

            optimizer.zero_grad()       # 反向传播和优化
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 计算epoch平均损失
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_epoch_loss}')

        # 检查是否是最低损失
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = model.state_dict()  # 保存模型状态
