import os
import numpy as np
import torch
import tiling
from torch.utils.data import Dataset
from torchvision import transforms


class TripletDataset(Dataset):
    def __init__(self, dataset_path):
        # change_proportions
        self.transforms = transforms.ToTensor()
        self.path = dataset_path
        with open(os.path.join(self.path, "change_proportions.npy"), "rb") as f:
            proportions = np.load(f)
        self.change_proportions = proportions

    def __len__(self):
        return int(
            (len([f for f in os.listdir(self.path) if f.endswith(".npy")]) - 1) / 3
        )

    def __getitem__(self, idx):
        with open(os.path.join(self.path, f"anchor_{idx}.npy"), "rb") as f:
            anchor = self.transforms(np.load(f))
        with open(os.path.join(self.path, f"positive_{idx}.npy"), "rb") as f:
            positive = self.transforms(np.load(f))
        with open(os.path.join(self.path, f"negative_{idx}.npy"), "rb") as f:
            negative = self.transforms(np.load(f))
        change_proportions = torch.tensor(self.change_proportions[idx])
        return anchor, positive, negative, change_proportions


class ValidTestDataset(Dataset):
    def __init__(self, before_images, after_images, change_masks):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.before_images = before_images  # list of list
        self.after_images = after_images
        self.change_masks = change_masks
        tile_size = 32
        self.tile_size = tile_size
        self.number_of_images_per_event = [len(images) for images in self.before_images]
        self.number_of_tiles_per_event = [
            self.get_number_of_tiles(image, tile_size) for image in after_images
        ]
        self.tile_counts = [
            image_number * number_of_tiles
            for image_number, number_of_tiles in zip(
                self.number_of_images_per_event, self.number_of_tiles_per_event
            )
        ]
        self.empty_before = 0
        self.empty_after = 0

    def __len__(self):
        return sum(self.tile_counts)

    def __getitem__(self, index):
        event_idx = self.get_event_idx(index)
        suma = sum(
            [
                tile_count
                for tile_count in self.tile_counts
                if self.tile_counts.index(tile_count) < event_idx
            ]
        )
        tile_idx = index - sum([self.tile_counts[i] for i in range(event_idx)])
        before_picture_idx = self.get_before_picture_idx(tile_idx, event_idx)
        tile_idx -= before_picture_idx * self.number_of_tiles_per_event[event_idx]
        mask = self.change_masks[event_idx]
        after_image = self.after_images[event_idx]
        before_image = self.before_images[event_idx][before_picture_idx]
        tile_height_index, tile_width_index = tiling.get_tile_height_and_width_indexes(
            after_image, tile_idx, self.tile_size
        )
        mask = tiling.get_tile(
            mask, tile_height_index, tile_width_index, self.tile_size, mask=True
        )
        after_image = tiling.get_tile(
            after_image, tile_height_index, tile_width_index, self.tile_size
        )
        before_image = tiling.get_tile(
            before_image, tile_height_index, tile_width_index, self.tile_size
        )
        empty_before = np.any(np.isnan(before_image))
        empty_after = np.any(np.isnan(after_image))
        after_image = self.transform(after_image)
        after_image = torch.nan_to_num(after_image)
        before_image = self.transform(before_image)
        before_image = torch.nan_to_num(before_image)
        return (
            mask,
            before_image,
            after_image,
            event_idx,
            before_picture_idx,
            tile_idx,
            empty_before,
            empty_after,
        )

    def get_number_of_tiles(self, image, tile_size):
        # format should be H, W, C
        assert image.shape[2] < 20
        width_num_tiles, height_num_tiles = tiling.get_width_and_height_number_of_tiles(
            image, tile_size
        )
        number_of_tiles = width_num_tiles * height_num_tiles
        return number_of_tiles

    def get_event_idx(self, tile_index):
        tile_sum = -1
        for event_index, tile_count in enumerate(self.tile_counts):
            tile_sum += tile_count
            if tile_index <= tile_sum:
                return event_index

    def get_before_picture_idx(self, tile_index, event_idx):
        before_picture_idx = tile_index // self.number_of_tiles_per_event[event_idx]
        return before_picture_idx

    def get_initialized_metrics(self):
        """return initialized metrics, shape is [event][tile][0][before_picture_id]
        and [event][tile][1] == portion of changed pixels"""
        initialized_metrics = []
        for event_id in range(len(self.after_images)):
            event_list = []
            for tile_id in range(self.number_of_tiles_per_event[event_id]):
                tile_list = []
                before_pictures_list = []
                for before_picture_id in range(
                    self.number_of_images_per_event[event_id]
                ):
                    before_pictures_list.append(0)
                tile_list.append(before_pictures_list)
                tile_list.append(0)
                event_list.append(tile_list)
            initialized_metrics.append(event_list)
        return initialized_metrics
