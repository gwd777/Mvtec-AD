import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

import contextlib
import logging
import sys
import numpy as np
import torch
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils


LOGGER = logging.getLogger(__name__)

_DATASETS = {"dataset": ["patchcore.datasets.mvtec", "MVTecDataset"]}


def run(
    results_path='results',
    gpu=(0,),
    seed=123,
    log_group='IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0',
    log_project='MVTecAD_Results',
    save_segmentation_images=False,
    save_patchcore_model=False,
):
    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = get_dataloaders(seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = get_sampler(device=device)
            PatchCore_list = get_patchcore(imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(dataloaders["testing"])
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


def get_patchcore(input_shape, sampler, device):
    backbone_names = ['wideresnet50']
    layers_to_extract_from = ('layer2', 'layer3')
    faiss_on_gpu = False
    faiss_num_workers = 8
    pretrain_embed_dimension = 1024
    target_embed_dimension = 1024
    patchsize = 3
    anomaly_scorer_num_nn = 1

    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    loaded_patchcores = []
    for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                backbone_name.split("-")[-1]
            )
        backbone = patchcore.backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

        patchcore_instance = patchcore.patchcore.PatchCore(device)
        patchcore_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            patchsize=patchsize,
            featuresampler=sampler,
            anomaly_scorer_num_nn=anomaly_scorer_num_nn,
            nn_method=nn_method,
        )
        loaded_patchcores.append(patchcore_instance)
    return loaded_patchcores


def get_sampler(name='approx_greedy_coreset', percentage=0.1, device=None):
    if name == "identity":
        return patchcore.sampler.IdentitySampler()
    elif name == "greedy_coreset":
        return patchcore.sampler.GreedyCoresetSampler(percentage, device)
    elif name == 'approx_greedy_coreset':
        return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)


def get_dataloaders(seed):
    name = 'dataset'
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    data_path = 'D:/mvtec_anomaly_detection'
    subdatasets = ['bottleeasy']
    resize = 256
    train_val_split = 1
    imagesize = 224
    augment = False
    num_workers = 8
    batch_size = 2

    dataloaders = []
    for subdataset in subdatasets:
        train_dataset = dataset_library.__dict__[dataset_info[1]](
            source=data_path,
            classname=subdataset,
            resize=resize,
            train_val_split=train_val_split,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=seed,
            augment=augment,
        )

        test_dataset = dataset_library.__dict__[dataset_info[1]](
            source=data_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TEST,
            seed=seed,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
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
