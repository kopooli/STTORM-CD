import os
import numpy as np
from tqdm import tqdm
import shutil


def extract_and_save_tiles(
    image_path,
    mask,
    most_recent_image_path,
    save_to,
    section,
    margin_size,
    stride,
    tile_size=32,
):
    min_change = margin_size[0]
    max_change = margin_size[1]
    date = image_path.split("/")[-1]
    date = date.split(".")[0]
    date = date.replace("-", "_")
    image = np.load(image_path).astype("float32")
    # input images are integers and can contain zero which is -inf when logged - causes issues in network, so here we are cliping to 1
    image = np.concatenate(
        (
            np.clip(image[:, :, :-1], 1, 100000, dtype="float32"),
            image[:, :, -1][:, :, np.newaxis],
        ),
        axis=-1,
    )
    # image shape H,W,C
    tile_pixels = pow(tile_size, 2)
    for width in range(0, image.shape[1] - tile_size, stride):
        for height in range(0, image.shape[0] - tile_size, stride):
            tile_mask = mask[height : height + tile_size, width : width + tile_size]
            clouds_mask = image[
                height : height + tile_size, width : width + tile_size, 13
            ]
            # clouds_mask = (clouds_mask > 40).astype(int)
            change_proportion = np.count_nonzero(tile_mask) / tile_pixels
            clouds_proportion = np.count_nonzero(clouds_mask) / tile_pixels
            date = date.split("_")
            date = f"{date.pop(0)}_{date.pop(0)}_{date.pop(0)}"
            if (
                clouds_proportion < 0.25
                and change_proportion >= min_change
                and change_proportion <= max_change
            ):
                tile = image[height : height + tile_size, width : width + tile_size, :]

                if image_path == most_recent_image_path:
                    tile_filename = f"changed_tile_section_{section}_height_{height}_width_{width}_{date}_{change_proportion}.npy"
                else:
                    tile_filename = f"unchanged_tile_section_{section}_height_{height}_width_{width}_{date}_{change_proportion}.npy"

                tile_filepath = save_to + tile_filename
                np.save(tile_filepath, tile)
    return


def find_most_recent_image(s2_folder):
    image_files = [f for f in os.listdir(s2_folder) if f.endswith(".npy")]
    image_files.sort(reverse=True)
    return os.path.join(s2_folder, image_files[0])


def find_change_mask(event_folder):
    mask = np.load(os.path.join(event_folder, "mask.npy"))
    mask = mask.astype(np.uint8)
    return mask


def tile_original_images(margin_size, stride):
    dataset_dir = "data/dataset/train"
    save_dir = f"data/tiled/prefinal"
    try:
        shutil.rmtree(save_dir)
    except:
        pass
    save_event = save_dir
    event_subfolders = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    for event_subfolder in tqdm(event_subfolders):
        event_path = os.path.join(dataset_dir, event_subfolder)
        sections_subfolders = [
            d
            for d in os.listdir(event_path)
            if os.path.isdir(os.path.join(event_path, d))
        ]
        for section in sections_subfolders:
            save_to = save_event + "/" + event_subfolder + section
            try:
                os.makedirs(save_to)
            except:
                pass
            save_to = save_to + "/"
            final_path = os.path.join(event_path, section)
            change_mask = find_change_mask(final_path)
            most_recent_image_path = find_most_recent_image(
                os.path.join(final_path, "all_bands")
            )

            s2_folder = os.path.join(final_path, "all_bands")
            for image_filename in [
                f for f in os.listdir(s2_folder) if f.endswith(".npy")
            ]:
                image_path = os.path.join(s2_folder, image_filename)
                extract_and_save_tiles(
                    image_path,
                    change_mask,
                    most_recent_image_path,
                    save_to,
                    section,
                    margin_size,
                    stride,
                )


if __name__ == "__main__":
    tile_original_images((0, 1), 32)
