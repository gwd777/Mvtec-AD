# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

import logging
import os
import sys

import PIL
import torch
import simplenet
from torchvision import transforms

sys.path.append("src")
import backbones
import utils

LOGGER = logging.getLogger(__name__)
_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
    # "hwhq": ["datasets.hwhq", "HWHQDataset"]
}
device = torch.device("cuda:0")

def run(
        results_path='results',
        gpu=None,
        seed=123,
        log_group='simplenet_mvtec',
        log_project='MVTecAD_Results',
        run_name='run',
        test=False,
        save_segmentation_images=False
):
    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    # list_of_dataloaders = get_dataloaders(name="mvtec", data_path='/sys/fs/cgroup/ccpd/mvtec/', subdatasets=['screw', 'pill'])
    # list_of_dataloaders = get_dataloaders(name="hwhq", data_path='D:\mvtec_anomaly_detection\hw_hq_data', subdatasets=['hw_hq_data', 'hw_ls_data'], resize=128)

    list_of_dataloaders = get_dataloaders(name="mvtec", data_path='D:\mvtec_anomaly_detection', subdatasets=['screw'], resize=64, batch_size=16, imagesize=64)
    # list_of_dataloaders = get_dataloaders(name="mvtec", data_path='D:\mvtec_anomaly_detection', subdatasets=['big_hw_ls_data'], resize=128, batch_size=20, ignore_mask=True)

    result_collect = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "_____________________Evaluating dataset [{}] ({}/{})______________________".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)
        dataset_name = dataloaders["training"].name

        imagesize = dataloaders["training"].dataset.imagesize
        # simplenet_list = methods["get_simplenet"](imagesize, device)

        simplenet_list = get_simplenet(input_shape=imagesize, device=device, meta_epochs=1)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, SimpleNet in enumerate(simplenet_list):
            # torch.cuda.empty_cache()
            if SimpleNet.backbone.seed is not None:
                utils.fix_seeds(SimpleNet.backbone.seed, device)

            LOGGER.info("Training models ({}/{})".format(i + 1, len(simplenet_list)))
            # torch.cuda.empty_cache()

            SimpleNet.set_model_dir(os.path.join(models_dir, f"{i}"), dataset_name)

            i_auroc, p_auroc, pro_auroc, acc = SimpleNet.train(dataloaders["training"], dataloaders["testing"])

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": i_auroc,      # auroc,
                    "full_pixel_auroc": p_auroc,    # full_pixel_auroc,
                    "anomaly_pixel_auroc": pro_auroc,
                    "acc": acc
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


def get_simplenet(
        backbone_names=['wideresnet50'] ,
        layers_to_extract_from=('layer2', 'layer3'),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        embedding_size=256,
        meta_epochs=2,
        aed_meta_epochs=1,
        gan_epochs=4,
        noise_std=0.015,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=.5,
        dsc_lr=0.0002,
        auto_noise=0,
        train_backbone=True,
        cos_lr=True,
        pre_proj=1,
        proj_layer_type=0,
        mix_noise=1,
        input_shape=None,
        device=None
):
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    simplenets = []
    for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
        backbone = backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        simplenet_inst = simplenet.SimpleNet(device)
        simplenet_inst.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            patchsize=patchsize,
            embedding_size=embedding_size,
            meta_epochs=meta_epochs,
            aed_meta_epochs=aed_meta_epochs,
            gan_epochs=gan_epochs,
            noise_std=noise_std,
            dsc_layers=dsc_layers,
            dsc_hidden=dsc_hidden,
            dsc_margin=dsc_margin,
            dsc_lr=dsc_lr,
            auto_noise=auto_noise,
            train_backbone=train_backbone,
            cos_lr=cos_lr,
            pre_proj=pre_proj,
            proj_layer_type=proj_layer_type,
            mix_noise=mix_noise,
        )
        simplenets.append(simplenet_inst)
    return simplenets


def get_dataloaders(
        name="mvtec",
        data_path='/sys/fs/cgroup/ccpd/mvtec',
        subdatasets=('screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor', 'metal_nut', 'bottle', 'hazelnut', 'leather'),
        train_val_split=1.0,
        batch_size=12,
        resize=329,
        imagesize=288,
        num_workers=4,
        rotate_degrees=0,
        translate=0,
        scale=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        gray=0.0,
        hflip=0.0,
        vflip=0.0,
        augment=True,
        ignore_mask=False,
        seed=123
):
    print('____subdatasets______>', subdatasets, data_path)

    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    dataloaders = []
    for subdataset in subdatasets:
        train_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=resize,
            train_val_split=train_val_split,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=seed,
            rotate_degrees=rotate_degrees,
            translate=translate,
            brightness_factor=brightness,
            contrast_factor=contrast,
            saturation_factor=saturation,
            gray_p=gray,
            h_flip_p=hflip,
            v_flip_p=vflip,
            scale=scale,
            augment=augment,
            ignore_mask=ignore_mask
        )

        test_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TEST,
            ignore_mask=ignore_mask,
            seed=seed,
        )

        LOGGER.info(f"____{subdataset}_____>Dataset: train={len(train_dataset)} test={len(test_dataset)}")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )

        train_dataloader.name = name
        if subdataset is not None:
            train_dataloader.name += "_" + subdataset

        if train_val_split < 1:
            val_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.VAL,
                seed=seed,
            )

            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=4,
                pin_memory=True,
            )
        else:
            val_dataloader = None

        dataloader_dict = {
            "training": train_dataloader,
            "validation": val_dataloader,
            "testing": test_dataloader,
        }

        dataloaders.append(dataloader_dict)
    return dataloaders


def getRange(vector):
    # 获取张量的最小值和最大值
    min_value = torch.min(vector)
    max_value = torch.max(vector)

    # 打印结果
    print(f"__________Min value: {min_value} / Max value: {max_value}")


def getNoise(true_feats):
    noise_idxs = torch.randint(0, 1, torch.Size([true_feats.shape[0]]))
    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=1).to(device)  # (N, K)
    noise = torch.stack([
        torch.normal(0, 0.05 * 1.1 ** (k), true_feats.shape)
        for k in range(1)], dim=1).to(device)  # (N, K, C)
    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
    return noise

def generateTensor(source, resize, imagesize, name='Tst'):
    transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.CenterCrop(imagesize),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    simplenet_list = get_simplenet(input_shape=(3, 288, 288), device=device, meta_epochs=1)
    simplenet = simplenet_list[0]

    train_elements_paths = os.listdir(source)
    print('________数据Len()_________>', len(train_elements_paths))
    data_to_iterate = []
    batch_num = 0
    for idx, elements_name in enumerate(train_elements_paths):
        image_path = os.path.join(source, elements_name)
        image = PIL.Image.open(image_path).convert("RGB")
        img_tenosr = transform_img(image).to(device)
        img_tenosr = img_tenosr.unsqueeze(0)
        getRange(img_tenosr)
        true_feats = simplenet._embed(img_tenosr, evaluation=True)[0]
        fake_feats = true_feats + getNoise(true_feats)

        # data_to_iterate.append((fake_feats, 1))
        # data_to_iterate.append((true_feats, 0))

        data_to_iterate.append((fake_feats.cpu().numpy(), 1))  # 转换为 ndarray
        data_to_iterate.append((true_feats.cpu().numpy(), 0))  # 转换为 ndarray

        if 300 / len(data_to_iterate) == 1:
            batch_num += 1
            torch.save(data_to_iterate, f'noise_{batch_num}_{name}_tensor.pt')
            data_to_iterate.clear()
            print('_______gen_______>', fake_feats.shape)
            break
    return data_to_iterate

def loadTensorData():
    loaded_data = torch.load(f'noise_500_1_tensor.pt')
    # 访问数据和标记
    for t_element in loaded_data:
        feature = t_element[0]
        is_anomaly = t_element[1]
        print(f"Data shape: {feature}, is_anomaly: {is_anomaly}")

def Tst():
    simplenet_list = get_simplenet(input_shape=(3, 288, 288), device=device, meta_epochs=1)
    simplenet = simplenet_list[0]
    tensor1 = torch.randn(1, 3, 288, 288)
    scores, segmentations, features = simplenet.predict(tensor1)
    tensor1 = tensor1.to(torch.float).to(device)
    true_feats = simplenet._embed(tensor1, evaluation=True)[0]
    # 用 [截断L1 计算损失] li=max(0,∣yi−y^i∣−δ)
    # scores = simplenet.discriminator(torch.cat([true_feats, fake_feats]))


def genTstTensor(isbad, source, resize, imagesize, name='Tst'):
    transform_img = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.CenterCrop(imagesize),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    simplenet_list = get_simplenet(input_shape=(3, 288, 288), device=device, meta_epochs=1)
    simplenet = simplenet_list[0]

    train_elements_paths = os.listdir(source)
    print('________数据Len()_________>', len(train_elements_paths))
    data_to_iterate = []
    batch_num = 0
    for idx, elements_name in enumerate(train_elements_paths):
        image_path = os.path.join(source, elements_name)
        image = PIL.Image.open(image_path).convert("RGB")
        img_tenosr = transform_img(image).to(device)
        img_tenosr = img_tenosr.unsqueeze(0)

        true_feats = simplenet._embed(img_tenosr, evaluation=True)[0]

        # 在第0维（最前面）添加一个新维度
        true_feats = true_feats.unsqueeze(0)

        data_to_iterate.append((true_feats, isbad))

        if 500 / len(data_to_iterate) == 1:
            batch_num += 1
            torch.save(data_to_iterate, f'noise_{batch_num}_{name}_tensor.pt')
            data_to_iterate.clear()
            print('_______gen_______>', true_feats.shape)

    torch.save(data_to_iterate, f'noise_{batch_num}_{name}_tensor.pt')
    print('________data_to_iterate_________>', len(data_to_iterate))
    return data_to_iterate



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    # run()

    generateTensor(source='D:\\mvtec_anomaly_detection\\big_hw_ls_data\\train\\good', resize=256, imagesize=288, name='Tst')

    # genTstTensor(isbad=0, source='D:\\mvtec_anomaly_detection\\big_hw_ls_data\\test\\good', resize=256, imagesize=288, name='Tst1000')

    # genTstTensor(isbad=1, source='D:\\mvtec_anomaly_detection\\big_hw_ls_data\\test\\anomaly_easy', resize=256, imagesize=288, name='TstEasy')

    # genTstTensor(isbad=1, source='D:\\mvtec_anomaly_detection\\big_hw_ls_data\\test\\anomaly_hard', resize=256, imagesize=288, name='TstHard')

    # loadTensorData()

    # Tst()


