import random
from random import sample
import argparse

import cv2
import numpy as np
import os
import pickle

from PIL.Image import Image
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, f1_score, fbeta_score
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

from datasets.mvtec import MVTecDataset

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('________当前设备_________>', device)

# set model's intermediate outputs
outputs = []
def hook(module, input, output):
    outputs.append(output)

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
        # self.discriminator = KAN([50176, 1024, 1])

    def forward(self, x):
        x = self.embeding(x)
        x = self.forward_modules["layer1"](x)
        x = self.forward_modules["layer2"](x)
        x = self.forward_modules["layer3"](x)
        # print('_______xLayer3______', x.shape)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # 展平
        x = self.discriminator(x)
        return x

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



def calculate_histogram2(image, bins=786):  # 786 / 256
    image = image.float()       # 确保Tensor是浮点型
    hist = torch.histc(image, bins=bins)    # 计算直方图
    hist = hist.numpy()         # 将直方图转换为numpy数组
    return hist
    
    
def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    # 将PIL直方图转换为OpenCV格式
    hist1 = np.float32(hist1)
    hist2 = np.float32(hist2)
    # 比较直方图
    return cv2.compareHist(hist1, hist2, method)


def calculate_dist_on_train(args, train_dataloader, model, class_name):
    train_outputs = OrderedDict([('layer1', []), ('layer2', [])])
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
        for layer_name in ['layer2']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
        # print('_______embedding_vectors________>', embedding_vectors.size())

        # 计算均值 Gram 矩阵
        BATCH_SIZE = embedding_vectors.size(0)  # torch.Size([140, 768, 56, 56])
        sum_hist = None
        for i in range(BATCH_SIZE):
            single_image = embedding_vectors[i: i + 1]
            hist_gray = calculate_histogram2(single_image)
            if sum_hist is None:
                sum_hist = hist_gray
            else:
                sum_hist += hist_gray
        mean_hist = sum_hist / BATCH_SIZE

        # save learned distribution
        train_outputs = [mean_hist]
        # train_outputs = [mean, cov]
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
    else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)
    return train_outputs


def calculate_dist_on_test(train_outputs, test_dataloader, model, class_name):
    test_outputs = OrderedDict([('layer1', []), ('layer2', [])])
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
    for layer_name in ['layer2']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    # 计算纹理损失得分
    hist_score_scores = []
    mean_hist = train_outputs[0]
    BATCH_SIZE = embedding_vectors.size(0)
    for i in range(BATCH_SIZE):
        single_image = embedding_vectors[i: i + 1]
        test_hist = calculate_histogram2(single_image)
        hist_score = compare_histograms(test_hist, mean_hist, method=cv2.HISTCMP_CORREL)
        hist_score_scores.append(hist_score)
    return gt_mask_list, gt_list, test_imgs, np.array(hist_score_scores)


def main():
    args = parse_args()

    # model = resnet18(pretrained=True, progress=True)
    model = SimpleCNN(wide_resnet50_2(pretrained=True, progress=True))
    best_model_state = torch.load('005_model_epoch_30.pth')
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    model.forward_modules["layer1"].register_forward_hook(hook)
    model.forward_modules["layer2"].register_forward_hook(hook)

    modelnull = SimpleCNN(wide_resnet50_2())
    modelnull.to(device)
    modelnull.eval()
    modelnull.forward_modules["layer1"].register_forward_hook(hook)
    modelnull.forward_modules["layer2"].register_forward_hook(hook)

    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    # model.layer1[-1].register_forward_hook(hook)
    # model.layer2[-1].register_forward_hook(hook)
    # model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:
        train_dataset = MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        # 加载训练均值
        train_outputs = calculate_dist_on_train(args, train_dataloader, model, class_name)
        
        gt_mask_list, gt_list, test_imgs, texture_loss_scores = calculate_dist_on_test(train_outputs, test_dataloader, modelnull, class_name)

        # Normalization
        max_score = texture_loss_scores.max()
        min_score = texture_loss_scores.min()
        scores = (texture_loss_scores - min_score) / (max_score - min_score)
        # print('_______sc________>', scores)
        # print('_______ts________>', texture_loss_scores)

        # calculate image-level ROC AUC score
        # img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, scores)  # 计算 ROC 曲线的数据点，可用于绘制 ROC 曲线
        img_roc_auc = roc_auc_score(gt_list, scores)  # 计算 ROC 曲线下的面积，用于评估二分类模型的整体性能。
        total_roc_auc.append(img_roc_auc)

        print('____%s______image ROCAUC: %.3f' % (class_name, img_roc_auc))
        # print('__________i___________>', len(scores), len(gt_list))
        # get_best_threshold_beta(gt_list, scores, class_name)
        get_best_threshold(gt_list, scores, class_name)
        
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1,
                 stride=s)  # (N, C, H, W)-->(N, C * kernel_size^2, k_h * k_w) [209, 256, 56, 56] ==> [209,256*2*2 784]
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        x_i = x[:, :, i, :, :]
        z[:, :, i, :, :] = torch.cat((x_i, y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)  # (N,CK,HW) --> (N, C, H, W)

    return z

def get_best_threshold(y_true, y_pred, class_name):
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # 初始化存储指标的列表
    accuracies = []
    recalls = []
    precision_scores = []
    f1_scores = []

    # 计算每个阈值对应的准确率、召回率和F1分数
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)

        accuracies.append(acc)
        recalls.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)

    # 找到最佳阈值（以F1分数为准）
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]

    # print(f'_____{class_name}_____Best Threshold: {best_threshold}')
    print(f'_____{class_name}_____Accuracy: {accuracies[best_threshold_index]}')
    print(f'_____{class_name}_____Recall: {recalls[best_threshold_index]}')
    print(f'_____{class_name}_____Precision: {precision_scores[best_threshold_index]}')
    print(f'_____{class_name}_____F1 Score: {f1_scores[best_threshold_index]}')


# 默认值 beta=2 表示 Recall 的权重是 Precision 的两倍。你可以根据需要调整 beta 的值   
def get_best_threshold_beta(y_true, y_pred, class_name, beta=2):
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # 初始化存储指标的列表
    accuracies = []
    recalls = []
    precision_scores = []
    fbeta_scores = []

    # 计算每个阈值对应的准确率、召回率和Fβ分数
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        fbeta = fbeta_score(y_true, y_pred_binary, beta=beta)

        accuracies.append(acc)
        recalls.append(recall)
        precision_scores.append(precision)
        fbeta_scores.append(fbeta)

    # 找到最佳阈值（以Fβ分数为准）
    best_threshold_index = np.argmax(fbeta_scores)
    best_threshold = thresholds[best_threshold_index]

    print(f'_____F{beta}_{class_name}_____Best Threshold: {best_threshold}')
    print(f'_____F{beta}_{class_name}_____Accuracy at Best Threshold: {accuracies[best_threshold_index]}')
    print(f'_____F{beta}_{class_name}_____Recall at Best Threshold: {recalls[best_threshold_index]}')
    print(f'_____F{beta}_{class_name}_____Precision at Best Threshold: {precision_scores[best_threshold_index]}')
    print(f'_____F{beta}_{class_name}_____ Score at Best Threshold: {fbeta_scores[best_threshold_index]}')

if __name__ == '__main__':
    main()
