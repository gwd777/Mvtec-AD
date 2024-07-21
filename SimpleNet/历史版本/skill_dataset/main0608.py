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
    parser.add_argument('--data_path', type=str, default='/home/wzx_python_project/')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


def calculate_histogram(image, is_color=True):
    if not is_color:
        image = image.convert('L')

    # 计算直方图
    if is_color:
        r, g, b = image.split()
        hist_r = np.array(r.histogram())
        hist_g = np.array(g.histogram())
        hist_b = np.array(b.histogram())
        return hist_r, hist_g, hist_b
    else:
        hist = np.array(image.histogram())
        return hist


def calculate_histogram2(image, bins=256):
    # 确保Tensor是浮点型
    image = image.float()

    # 计算直方图
    hist, _ = torch.histc(image, bins=bins)

    # 将直方图转换为numpy数组
    hist = hist.numpy()

    return hist
def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    # 将PIL直方图转换为OpenCV格式
    hist1 = np.float32(hist1)
    hist2 = np.float32(hist2)

    # 比较直方图
    return cv2.compareHist(hist1, hist2, method)

def calculate_dist_on_train(args, train_dataloader, model, class_name, idx):
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
        print('_______embedding_vectors________>', embedding_vectors.size())

        # 计算均值 Gram 矩阵
        BATCH_SIZE = embedding_vectors.size(0)
        sum_hist = None
        for i in range(BATCH_SIZE):
            single_image = embedding_vectors[i: i + 1]
            hist_gray = calculate_histogram(single_image, is_color=False)
            if sum_hist is None:
                sum_hist = hist_gray
            else:
                sum_hist += hist_gray
        mean_hist = sum_hist / BATCH_SIZE

        # save learned distribution
        train_outputs = [mean_hist.numpy()]
        # train_outputs = [mean, cov]
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
    else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)
    return train_outputs


def calculate_dist_on_test(train_outputs, test_dataloader, model, class_name, idx):
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
        test_hist = calculate_histogram(single_image, is_color=False)
        hist_score = compare_histograms(test_hist, mean_hist, method=cv2.HISTCMP_CORREL)
        hist_score_scores.append(hist_score)
    return gt_mask_list, gt_list, test_imgs, np.array(hist_score_scores)


def main():
    args = parse_args()

    # model = resnet18(pretrained=True, progress=True)
    resnet = wide_resnet50_2(pretrained=True, progress=True)
    
    # 加载参数
    from features_extractor0608 import SimpleCNN
    model = SimpleCNN(resnet)
    best_model_state = torch.load('002_best_model_big_hw_ls_1000data_easy.pth')
    
    # 打印状态字典中的键
    # for key in best_model_state.keys(): print('_________SCNN-Key_____________>', key)

    model.load_state_dict(best_model_state)

    t_d = 1792
    d = 550
        
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

    model.forward_modules["layer1"].register_forward_hook(hook)
    model.forward_modules["layer2"].register_forward_hook(hook)
    
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
        train_outputs = calculate_dist_on_train(args, train_dataloader, model, class_name, idx)
        
        gt_mask_list, gt_list, test_imgs, texture_loss_scores = calculate_dist_on_test(train_outputs, test_dataloader, resnet, class_name, idx)

        # Normalization
        max_score = texture_loss_scores.max()
        min_score = texture_loss_scores.min()
        scores = (texture_loss_scores - min_score) / (max_score - min_score)
        print('_______sc________>', scores)
        print('_______ts________>', texture_loss_scores)
        
        
        # calculate image-level ROC AUC score
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, scores)  # 计算 ROC 曲线的数据点，可用于绘制 ROC 曲线
        img_roc_auc = roc_auc_score(gt_list, scores)  # 计算 ROC 曲线下的面积，用于评估二分类模型的整体性能。
        total_roc_auc.append(img_roc_auc)

        print('__________i___________>', len(scores), len(gt_list))
        get_best_threshold_beta(gt_list, scores, class_name)
        get_best_threshold(gt_list, scores, class_name)
        print('__________image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

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

    print(f'_____{class_name}_____Best Threshold: {best_threshold}')
    print(f'_____{class_name}_____Accuracy at Best Threshold: {accuracies[best_threshold_index]}')
    print(f'_____{class_name}_____Recall at Best Threshold: {recalls[best_threshold_index]}')
    print(f'_____{class_name}_____Precision at Best Threshold: {precision_scores[best_threshold_index]}')
    print(f'_____{class_name}_____F1 Score at Best Threshold: {f1_scores[best_threshold_index]}')


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
