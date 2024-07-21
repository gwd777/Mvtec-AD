# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

import logging
import os
import sys

import torch

sys.path.append("src")
import backbones
from 历史版本 import simplenet0417
import utils

LOGGER = logging.getLogger(__name__)
_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
}


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
    # pid = os.getpid()
    # list_of_dataloaders = methods["get_dataloaders"](seed)

    list_of_dataloaders = get_dataloaders(name="mvtec")
    # device = utils.set_torch_device(gpu)
    device = torch.device("cuda:0")

    result_collect = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        imagesize = dataloaders["training"].dataset.imagesize
        # simplenet_list = methods["get_simplenet"](imagesize, device)
        simplenet_list = get_simplenet(input_shape=imagesize, device=device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, SimpleNet in enumerate(simplenet_list):
            # torch.cuda.empty_cache()
            if SimpleNet.backbone.seed is not None:
                utils.fix_seeds(SimpleNet.backbone.seed, device)
            LOGGER.info(
                "Training models ({}/{})".format(i + 1, len(simplenet_list))
            )
            # torch.cuda.empty_cache()

            SimpleNet.set_model_dir(os.path.join(models_dir, f"{i}"), dataset_name)
            if not test:
                i_auroc, p_auroc, pro_auroc = SimpleNet.train(dataloaders["training"], dataloaders["testing"])
            else:
                # BUG: the following line is not using. Set test with True by default.
                # i_auroc, p_auroc, pro_auroc =  SimpleNet.test(dataloaders["training"], dataloaders["testing"], save_segmentation_images)
                print("Warning: Pls set test with true by default")

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": i_auroc,  # auroc,
                    "full_pixel_auroc": p_auroc,  # full_pixel_auroc,
                    "anomaly_pixel_auroc": pro_auroc,
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
        meta_epochs=4,
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
        data_path='D:\mvtec_anomaly_detection',
        # subdatasets=('screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor', 'metal_nut', 'bottle', 'hazelnut', 'leather'),
        subdatasets=['screw'],
        train_val_split=1.0,
        batch_size=12,
        # resize=329,
        resize=256,
        imagesize=288,
        num_workers=2,
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
        seed=123
):
    print('____subdatasets______>', subdatasets)

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
        )

        test_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TEST,
            seed=seed,
        )

        LOGGER.info(f"Dataset: train={len(train_dataset)} test={len(test_dataset)}")

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    run()