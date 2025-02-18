import tifffile as tiff
import os
import numpy as np
from utils import cloud_detector
from tqdm import tqdm


def generate_single_cloud_mask(picture_path, cloud_detector):
    picture = tiff.imread(picture_path)
    picture_for_clouds = picture.copy() / 10000
    cloud_mask = cloud_detector.get_cloud_masks(picture_for_clouds[np.newaxis, ...])
    print(cloud_mask)
    return picture, cloud_mask


def generate_single_cloud_mask_for_image(picture, cloud_detector):
    picture_for_clouds = picture.copy() / 10000
    cloud_mask = cloud_detector.get_cloud_masks(picture_for_clouds[np.newaxis, ...])
    print(cloud_mask)
    return picture, cloud_mask


def append_probability_map_to_image(picture, cloud_mask):
    cloud_mask = cloud_mask.transpose((1, 2, 0))
    new_picture = np.concatenate((picture, cloud_mask), axis=2)
    return new_picture


def save_new_npys(sector_path):
    tiff_files_path = os.path.join(sector_path, "all_bands")
    for file in tqdm(os.listdir(tiff_files_path)):
        if ".tiff" not in file:
            continue
        path = os.path.join(tiff_files_path, file)
        picture, cloud_mask = generate_single_cloud_mask(path, cloud_detector)
        new_picture = append_probability_map_to_image(picture, cloud_mask)
        new_file = file.replace("tiff", "npy")
        np.save(os.path.join(tiff_files_path, new_file), new_picture)
        os.remove(os.path.join(tiff_files_path, file))


dataset_path = "data/dataset"
for dataset_type in [
    d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
]:
    type_path = os.path.join(dataset_path, dataset_type)
    for event in [
        d for d in os.listdir(type_path) if os.path.isdir(os.path.join(type_path, d))
    ]:
        event_path = os.path.join(type_path, event)
        if dataset_type == "train":
            for sector in [
                d
                for d in os.listdir(event_path)
                if os.path.isdir(os.path.join(event_path, d))
            ]:
                sector_path = os.path.join(event_path, sector)
                save_new_npys(sector_path)
        # test
        else:
            save_new_npys(event_path)
