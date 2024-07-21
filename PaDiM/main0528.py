import random
from random import sample
import argparse
import numpy as np
import os
import pickle

from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('________当前设备_________>', device)

# set model's intermediate outputs
outputs = []


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
        loss = F.mse_loss(G_output, G_target)
        return loss

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='D:/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()

def calculate_dist_on_train(args, train_dataloader, model, class_name, idx):
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    global outputs

    # extract train set features
    train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
    if not os.path.exists(train_feature_filepath):
        for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
            break

        # randomly select d dimension
        embedding_vectors_select = torch.index_select(embedding_vectors, 1, idx)

        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors_select.size()
        embedding_vectors_select = embedding_vectors_select.view(B, C, H * W)
        mean = torch.mean(embedding_vectors_select, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        for i in range(H * W):
            embedv = embedding_vectors_select[:, :, i].numpy()
            cov[:, :, i] = np.cov(embedv, rowvar=False) + 0.01 * I  # 用于计算协方差矩阵

        # save learned distribution
        train_outputs = [mean, cov]
        # train_outputs = [mean, cov]
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
    else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)
    return train_outputs

def calculate_dist_on_test(train_outputs, test_dataloader, model, class_name, idx):
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    gt_list = []
    gt_mask_list = []
    test_imgs = []
    global outputs

    # extract test set features
    for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
        # initialize hook outputs
        outputs = []
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
        break

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)  # 张量的第 1 维（即通道维度）中选择特定的维度。idx 是包含要选择的维度索引的张量

    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()  # (B, C, H, W) --> (B, C, HW) 这是为了将二维的图像展平成一维

    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]  # 获取位置 i 处的均值向量
        train_v = train_outputs[1][:, :, i]  # 获取位置 i 处的协方差矩阵。
        conv_inv = np.linalg.inv(train_v)  # 计算协方差矩阵的逆矩阵
        # dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist = []
        for samplex in embedding_vectors:
            sample_i = samplex[:, i]  # 获取样本 samplex 在位置 i 处的向量
            distance = mahalanobis(sample_i, mean, conv_inv)  # 计算样本 sample_i 到均值向量 mean 的马氏距离
            dist.append(distance)  # 将计算出的距离 distance 添加到 dist 列表中
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample unsqueeze(1) 在第 1 维增加一个维度 (B, H * W)-->(B, 1, H * W); F.interpolate 函数对输入张量进行插值，调整其空间尺寸; squeeze() 移除所有为 1 的维度
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):  # score_map 是一个包含多张得分图的 NumPy 数组。score_map.shape[0] 是批量大小（即有多少张得分图）。这行代码遍历每一张得分图
        score_map[i] = gaussian_filter(score_map[i], sigma=4)  # 对第 i 张得分图应用高斯滤波; sigma=4 是高斯滤波的标准差，决定了滤波的平滑程度。较大的 sigma 值会产生更平滑的结果 ; 高斯滤波是一种卷积操作，用一个高斯核（Gaussian kernel）对图像进行平滑处理，消除噪声和细节

    return score_map, dist_list, gt_mask_list, gt_list, test_imgs


def main():
    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 768
        d = 310
    model.to(device)

    # print(model)

    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        train_outputs = calculate_dist_on_train(args, train_dataloader, model, class_name, idx)
        score_map, dist_list, gt_mask_list, gt_list, test_imgs= calculate_dist_on_test(train_outputs, test_dataloader, model, class_name, idx)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)    # 每个样本的所有得分展平成一行，然后计算每个样本的最大得分
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)        # 计算 ROC 曲线的数据点，可用于绘制 ROC 曲线
        img_roc_auc = roc_auc_score(gt_list, img_scores)    # 计算 ROC 曲线下的面积，用于评估二分类模型的整体性能。
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())     # 计算精确率-召回率曲线的数据点，可用于绘制精确率-召回率曲线
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)    # (N, C, H, W)-->(N, C * kernel_size^2, k_h * k_w) [209, 256, 56, 56] ==> [209,256*2*2 784]
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        x_i = x[:, :, i, :, :]
        z[:, :, i, :, :] = torch.cat((x_i, y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)    # (N,CK,HW) --> (N, C, H, W)

    return z


if __name__ == '__main__':
    main()
