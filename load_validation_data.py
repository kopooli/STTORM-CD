import os
import numpy as np
import datetime
from preprocess_train_tiles import normalize_band
import tifffile as tiff
from tqdm import tqdm


def get_validation_data(ravaen, disaster_type):
    # loads ravaen test dataset (valid)
    if ravaen:
        root_folder = f"./data/ravaen/{disaster_type}"
    # loads my test dataset (test)
    else:
        root_folder = "data/dataset/test"
    before_images = []
    after_images = []
    change_masks = []
    for event in os.listdir(root_folder):
        if not os.path.isdir(os.path.join(root_folder, event)):
            continue
        event_path = os.path.join(root_folder, event)
        if ravaen:
            images_folder = os.path.join(event_path, "S2")
            change_mask_name = [
                f
                for f in os.listdir(os.path.join(event_path, "changes"))
                if f.endswith(".tif")
            ][0]
            change_mask_path = os.path.join(event_path, "changes", change_mask_name)
            change_mask = tiff.imread(change_mask_path)
            ravaen_cloud_mask = (change_mask == 2).astype(int)
            change_mask[(change_mask == 2)] = 0
            suffix = ".tif"
        else:
            images_folder = os.path.join(event_path, "all_bands")
            change_mask_name = "mask.npy"
            change_mask_path = os.path.join(event_path, change_mask_name)
            change_mask = np.load(change_mask_path)
            suffix = ".npy"
        change_masks.append(change_mask)
        images = os.listdir(images_folder)
        images = [(image_name.split("-"), image_name) for image_name in images]
        images = [
            (
                [
                    int(splitted[0]),
                    int(splitted[1]),
                    int(splitted[2].replace(suffix, "")),
                ],
                image_name,
            )
            for splitted, image_name in images
        ]
        images = [
            (
                datetime.date(year=splitted[0], month=splitted[1], day=splitted[2]),
                image_name,
            )
            for splitted, image_name in images
        ]
        images.sort(key=lambda x: x[0])
        temp_after_image = images.pop(-1)
        if ravaen:
            temp_after_image = tiff.imread(
                os.path.join(images_folder, temp_after_image[1])
            )[:, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14]].astype("float32")
            temp_after_image[:, :, -1] = ravaen_cloud_mask
        else:
            temp_after_image = np.load(
                os.path.join(images_folder, temp_after_image[1])
            )[:, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]].astype("float32")
        temp_after_image[:, :, :10] = np.clip(
            temp_after_image[:, :, :10], 1, 100000, dtype="float32"
        )
        temp_before_images = images
        if ravaen:
            temp_before_images = [
                tiff.imread(os.path.join(images_folder, image_name))[
                    :, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14]
                ].astype("float32")
                for splitted, image_name in temp_before_images
            ]
        else:
            temp_before_images = [
                np.load(os.path.join(images_folder, image_name))[
                    :, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]
                ].astype("float32")
                for splitted, image_name in temp_before_images
            ]
        temp_before_images = [
            np.concatenate(
                (
                    np.clip(bfr_img[:, :, :10], 1, 100000, dtype="float32"),
                    bfr_img[:, :, 10][:, :, np.newaxis],
                ),
                axis=-1,
            )
            for bfr_img in temp_before_images
        ]
        for channel in tqdm(range(11)):
            if channel != 10:
                temp_after_image = normalize_band(temp_after_image, channel)
            for i in range(len(temp_before_images)):
                if channel == 10:
                    if ravaen:
                        temp_before_images[i][:, :, channel] = (
                            temp_before_images[i][:, :, channel] > 40
                        ).astype(int)
                else:
                    temp_before_images[i] = normalize_band(
                        temp_before_images[i], channel
                    )
        before_images.append(temp_before_images)
        after_images.append(temp_after_image)
    return before_images, after_images, change_masks
