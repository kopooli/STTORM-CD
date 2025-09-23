import os
import numpy as np
import datetime
from preprocess_train_tiles import normalize_band
import tifffile as tiff
from tqdm import tqdm


def get_validation_data(ravaen, disaster_type):
    """
    Loads validation data, handling both RaVAEn's TIFF files and a custom dataset's NumPy files.

    Args:
        ravaen (bool): True if loading the RaVAEn dataset, False for the custom dataset.
        disaster_type (str): The specific disaster event to load from the RaVAEn dataset.

    Returns:
        tuple: A tuple containing three lists: before_images, after_images, and change_masks.
    """
    # Determine the root folder based on the dataset type
    if ravaen:
        root_folder = f"./data/ravaen/{disaster_type}"
    else:
        root_folder = "data/dataset/test"

    before_images = []
    after_images = []
    change_masks = []

    # Iterate through each event directory in the root folder
    for event in os.listdir(root_folder):
        event_path = os.path.join(root_folder, event)

        # Skip non-directory files
        if not os.path.isdir(event_path):
            continue

        images_folder = os.path.join(event_path, "S2" if ravaen else "all_bands")

        # Load change mask and determine file suffix
        if ravaen:
            change_mask_name = [
                f
                for f in os.listdir(os.path.join(event_path, "changes"))
                if f.endswith(".tif")
            ][0]
            change_mask_path = os.path.join(event_path, "changes", change_mask_name)
            change_mask = tiff.imread(change_mask_path)
            ravaen_cloud_mask = (change_mask == 2).astype(int)
            change_mask[(change_mask == 2)] = 0  # Ignore RaVAEn cloud mask
            suffix = ".tif"
        else:
            change_mask_path = os.path.join(event_path, "mask.npy")
            change_mask = np.load(change_mask_path)
            suffix = ".npy"

        change_masks.append(change_mask)

        # Get and sort image files by date
        images = os.listdir(images_folder)
        images = [f for f in images if f.endswith(suffix)]  # Filter for correct file type
        images_with_dates = []
        for image_name in images:
            # Assumes file names are in YYYY-MM-DD format
            date_str = image_name.split("-")
            date_obj = datetime.date(
                year=int(date_str[0]),
                month=int(date_str[1]),
                day=int(date_str[2].replace(suffix, "")),
            )
            images_with_dates.append((date_obj, image_name))
        
        images_with_dates.sort(key=lambda x: x[0])

        # Separate the "after" image (last one) from the "before" images
        temp_after_image_name = images_with_dates.pop(-1)[1]

        # Load and preprocess the "after" image
        if ravaen:
            temp_after_image = tiff.imread(os.path.join(images_folder, temp_after_image_name))
            temp_after_image = temp_after_image[:, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14]].astype("float32")
            temp_after_image[:, :, -1] = ravaen_cloud_mask  # Add cloud mask
        else:
            temp_after_image = np.load(os.path.join(images_folder, temp_after_image_name))
            temp_after_image = temp_after_image[:, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]].astype("float32")

        temp_after_image[:, :, :10] = np.clip(
            temp_after_image[:, :, :10], 1, 100000, dtype="float32"
        )
        
        # Load and preprocess all "before" images
        temp_before_images = []
        for _, image_name in images_with_dates:
            if ravaen:
                bfr_img = tiff.imread(os.path.join(images_folder, image_name))
                bfr_img = bfr_img[:, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14]].astype("float32")
                # Handle RaVAEn cloud mask
                bfr_img[:, :, -1] = (bfr_img[:, :, -1] > 40).astype(int)
            else:
                bfr_img = np.load(os.path.join(images_folder, image_name))
                bfr_img = bfr_img[:, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]].astype("float32")

            # Clip and add to the list
            bfr_img[:, :, :10] = np.clip(bfr_img[:, :, :10], 1, 100000, dtype="float32")
            temp_before_images.append(bfr_img)
        
        # Normalize bands for both before and after images
        for channel in tqdm(range(11)):
            if channel != 10:  # Skip the cloud mask channel (channel 10)
                temp_after_image = normalize_band(temp_after_image, channel)
                for i in range(len(temp_before_images)):
                    temp_before_images[i] = normalize_band(temp_before_images[i], channel)

        before_images.append(temp_before_images)
        after_images.append(temp_after_image)

    return before_images, after_images, change_masks