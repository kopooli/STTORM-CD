import numpy as np
import os
import datetime
from tqdm import tqdm
from random import randrange
import shutil

band_config = {
    0: {"x0": 6.9, "x1": 7.5, "y0": -1, "y1": 1},
    1: {"x0": 6.5, "x1": 7.4, "y0": -1, "y1": 1},
    2: {"x0": 6.2, "x1": 7.5, "y0": -1, "y1": 1},
    3: {"x0": 6.1, "x1": 7.5, "y0": -1, "y1": 1},
    4: {"x0": 6.5, "x1": 8, "y0": -1, "y1": 1},
    5: {"x0": 6.5, "x1": 8, "y0": -1, "y1": 1},
    6: {"x0": 6.5, "x1": 8, "y0": -1, "y1": 1},
    7: {"x0": 6.5, "x1": 8, "y0": -1, "y1": 1},
    8: {"x0": 6, "x1": 8, "y0": -1, "y1": 1},
    9: {"x0": 6, "x1": 8, "y0": -1, "y1": 1},
}


def normalize_band(input_array, channel_idx):
    input = np.log(input_array[:, :, channel_idx])
    output = (
        2
        * (
            (input - band_config[channel_idx]["x0"])
            / (band_config[channel_idx]["x1"] - band_config[channel_idx]["x0"])
        )
        - 1
    )
    input_array[:, :, channel_idx] = output
    return input_array


def unnormalize_tile(preprocessed_tile):
    preprocessed_tile = preprocessed_tile + 1
    preprocessed_tile = preprocessed_tile / 2
    for band_id in range(10):
        preprocessed_tile[:, band_id, :, :] = (
            preprocessed_tile[:, band_id, :, :]
            * (band_config[band_id]["x1"] - band_config[band_id]["x0"])
            + band_config[band_id]["x0"]
        )
    original_tile = np.exp(preprocessed_tile)
    return original_tile


changed_prefix = "changed_tile_"
unchanged_prefix = "unchanged_tile_"


def get_anchor_positive_negative_tiles(destination_path):
    try:
        shutil.rmtree(destination_path)
    except:
        pass
    try:
        os.makedirs(destination_path)
    except:
        pass
    idx = 0
    change_proportions = []
    path = "data/tiled/prefinal"
    for event in [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]:
        event_path = os.path.join(path, event)
        unchanged_event_pictures = [
            f for f in os.listdir(event_path) if "unchanged" in f and ".npy" in f
        ]
        set_unchanged = set(unchanged_event_pictures)
        changed_event_pictures = [
            f for f in os.listdir(event_path) if f not in set_unchanged and ".npy" in f
        ]
        unchanged_event_names = [
            f.split("tile_")[1].replace(".npy", "") for f in unchanged_event_pictures
        ]
        unchanged_event_names = set(unchanged_event_names)
        changed_event_names = [
            f.split("tile_")[1].replace(".npy", "") for f in changed_event_pictures
        ]
        for change_name in tqdm(changed_event_names):
            splitted = change_name.split("_")
            section_string = splitted.pop(0)
            section = splitted.pop(0)
            height_string = splitted.pop(0)
            height = splitted.pop(0)
            width_string = splitted.pop(0)
            width = splitted.pop(0)
            change_proportion = splitted.pop(-1)
            related_unchanged_pictures = set(
                [
                    f
                    for f in unchanged_event_names
                    if f"height_{height}_width_{width}_" in f and change_proportion in f
                ]
            )
            changed_date = datetime.date(
                year=int(splitted[0]), month=int(splitted[1]), day=int(splitted[2])
            )
            unchanged_dates = [
                datetime.date(
                    year=int(f.split("_")[6]),
                    month=int(f.split("_")[7]),
                    day=int(f.split("_")[8]),
                )
                for f in related_unchanged_pictures
                if f != change_name
            ]
            unchanged_dates.sort()
            for anchor_index in range(len(unchanged_dates)):
                for positive_index in range(anchor_index + 1, len(unchanged_dates)):
                    anchor_date = unchanged_dates[anchor_index]
                    positive_date = unchanged_dates[positive_index]
                    if positive_date == anchor_date:
                        continue

                    anchor_name = f"{unchanged_prefix}section_{section}_height_{height}_width_{width}_{anchor_date.year}_{str(anchor_date.month).zfill(2)}_{str(anchor_date.day).zfill(2)}_{change_proportion}.npy"
                    positive_name = f"{unchanged_prefix}section_{section}_height_{height}_width_{width}_{positive_date.year}_{str(positive_date.month).zfill(2)}_{str(positive_date.day).zfill(2)}_{change_proportion}.npy"
                    negative_name = f"{changed_prefix}{change_name}.npy"

                    anchor_image_0 = np.load(os.path.join(event_path, anchor_name))
                    positive_image_0 = np.load(os.path.join(event_path, positive_name))
                    negative_image_0 = np.load(os.path.join(event_path, negative_name))
                    anchor_image = anchor_image_0[
                        :, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
                    ]
                    positive_image = positive_image_0[
                        :, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
                    ]
                    negative_image = negative_image_0[
                        :, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
                    ]
                    for channel_idx in range(10):
                        anchor_image = normalize_band(anchor_image, channel_idx)
                        positive_image = normalize_band(positive_image, channel_idx)
                        negative_image = normalize_band(negative_image, channel_idx)
                    if any(
                        [
                            np.isinf(anchor_image).any(),
                            np.isinf(positive_image).any(),
                            np.isinf(negative_image).any(),
                        ]
                    ):
                        print("inf")
                        continue
                    if any(
                        [
                            np.isnan(anchor_image).any(),
                            np.isnan(positive_image).any(),
                            np.isnan(negative_image).any(),
                        ]
                    ):
                        print("nan")
                        continue
                    np.save(
                        os.path.join(destination_path, f"anchor_{idx}.npy"),
                        anchor_image,
                    )
                    np.save(
                        os.path.join(destination_path, f"positive_{idx}.npy"),
                        positive_image,
                    )
                    np.save(
                        os.path.join(destination_path, f"negative_{idx}.npy"),
                        negative_image,
                    )
                    change_proportions.append(float(change_proportion))
                    idx += 1
    np.save(
        file=(os.path.join(destination_path, "change_proportions.npy")),
        arr=np.array(change_proportions),
    )


if __name__ == "__main__":
    get_anchor_positive_negative_tiles("data/tiled/final")
