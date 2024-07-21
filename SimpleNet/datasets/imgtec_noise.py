import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

device = torch.device("cuda:0")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class ImgNoiseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source,
            resize=128,
            imagesize=288,
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.index_anomaly = []
        self.index_normal = []
        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            # transforms.RandomHorizontalFlip(h_flip_p),
            # transforms.RandomVerticalFlip(v_flip_p),
            # transforms.RandomGrayscale(gray_p),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        # 加载数据集
        self.data_to_iterate = self.get_image_data()

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        classname, anomaly, image_path, is_anomaly, inx = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        image_ndarray = image.numpy()
        return {
            "inx": inx,
            "image": image,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": is_anomaly,
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_ndarray": image_ndarray,
        }

    # 合成噪声，添加噪声
    def get_noise(self, true_feats):
        noise_idxs = torch.randint(0, 1, torch.Size([true_feats.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=1).to(device)  # (N, K)
        noise = torch.stack([
            torch.normal(0, 0.05 * 1.1 ** (k), true_feats.shape)
            for k in range(1)], dim=1).to(device)  # (N, K, C)
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        return noise

    def get_image_data(self):
        train_elements_paths = os.listdir(self.source)
        data_to_iterate = []
        for elements_name in train_elements_paths:
            image_path = os.path.join(self.source, elements_name)
            image = PIL.Image.open(image_path).convert("RGB")
            image = self.transform_img(image)
            image_ndarray = image.numpy()
            data_to_iterate.append(image_ndarray)
        return data_to_iterate

if __name__ == '__main__':
    train_dataset = ImgNoiseDataset(source='D:\mvtec_anomaly_detection\\big_hw_ls_data\\train\\good')

    for index, data_item in enumerate(train_dataset):
        img = data_item["image"]
        label = data_item['is_anomaly']
        # img = img.to(torch.float).to(device)
        print(index, img.shape, label)