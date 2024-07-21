import torch
from torch import nn
import torch.nn.functional as F

def gram_matrix(feature_map):
    (b, ch, h, w) = feature_map.size()
    features = feature_map.view(b, ch, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(ch * h * w)

class TextureLoss(nn.Module):
    def __init__(self):
        super(TextureLoss, self).__init__()

    def forward(self, output, target):
        G_output = gram_matrix(output)
        G_target = gram_matrix(target)
        return F.mse_loss(G_output, G_target)

# 使用纹理损失进行训练
texture_criterion = TextureLoss()


# ===================训练模型============================
num_epochs = 15
model,dataloader,optimizer,criterion = None
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)

        # 分类损失
        classification_loss = criterion(outputs, labels)

        # 纹理损失
        texture_loss = texture_criterion(outputs, images)

        # 总损失
        loss = classification_loss + texture_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')

