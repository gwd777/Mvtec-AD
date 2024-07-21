import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
from tqdm import tqdm
from datasets import mvtec
from kan import KAN

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

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=50176, out_features=1024, bias=True),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=1, bias=True)
        )
        
        self.discriminator_kan = KAN([50176, 1024, 1])

    def forward(self, x):
        x = self.embeding(x)
        x = self.forward_modules["layer1"](x)
        x = self.forward_modules["layer2"](x)
        x = self.forward_modules["layer3"](x)
        # print('_______xLayer3______', x.shape)  _______xLayer3______ torch.Size([40, 1024, 14, 14])
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # 展平
        x = self.discriminator_kan(x)
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

# 计算纹理损失
def gram_matrix(feature_map):
    (b, ch, h, w) = feature_map.size()
    features = feature_map.view(b, ch, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(ch * h * w)
    
def texture_loss(output, target):
    G_output = gram_matrix(output)
    G_target = gram_matrix(target)
    loss = F.mse_loss(G_output, G_target)
    return loss
        
if __name__ == '__main__':
    resnet = wide_resnet50_2(pretrained=True, progress=True)
    model = SimpleCNN(resnet)
    model.train()
    model.to(device)

    # 定义优化器
    dsc_optim = torch.optim.Adam(model.parameters(), lr=0.000007)   # , weight_decay=1e-5
    # dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(dsc_optim, 30, 0.0003)
        
    data_path = 'D:/mvtec_anomaly_detection'
    for class_name in mvtec.CLASS_NAMES:
        train_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, pin_memory=True)
        epoch_loss = []
        EPOCH_NUM = 20
        best_loss = float('inf')  # 初始化为正无穷大
        for epoch in range(EPOCH_NUM):  # 假设训练10个epoch
            for (img, _, _) in tqdm(train_dataloader, '| feature trainModel | train | %s |' % class_name):

                img = img.to(device)

                # 添加随机噪声生成负样本（每次都不同）
                T_noisy = add_gaussian_noise(img).to(device)

                # 前向传播
                T_features_scores = model(img)
                T_noisy_features_scores = model(T_noisy)

                # 计算 KL 散度损失
                # kl_loss = kl_divergence(T_features, T_noisy_features)
                # text_loss = texture_loss(T_features, T_noisy_features)
                # print(f'___________txtLoss={text_loss} / KLLoss={kl_loss}__________>')
                # loss = -(kl_loss + text_loss)

                true_loss = torch.clip(-T_features_scores + 3, min=0)
                fake_loss = torch.clip(T_noisy_features_scores + 3, min=0)
                loss = true_loss.mean() + fake_loss.mean()

                # 反向传播和优化
                dsc_optim.zero_grad()
                loss.backward()
                dsc_optim.step()

                # 更新学习率
                # dsc_schl.step()

                epoch_loss.append(loss)

            avg_loss = sum(epoch_loss) / len(epoch_loss)
            print(f'________{class_name}__________>Epoch [{epoch + 1}/20], Loss: {avg_loss}')

            if avg_loss < best_loss:
              best_loss = avg_loss
              best_model_state = model.state_dict()  # 保存模型状态
        
     
    # 在所有 epoch 完成后，保存损失最低的模型
    torch.save(best_model_state, f'006_model_epoch_{EPOCH_NUM}.pth')
    print(f'Best model saved with loss: {best_loss}')
   
        
        
