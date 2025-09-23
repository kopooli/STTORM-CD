import torch
import copy
import tiling
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor, optim
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np
from statistics import mean
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve, f1_score, average_precision_score
from preprocess_train_tiles import unnormalize_tile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from typing import List, Any, Dict, Tuple
# To create heatmaps, do it for each method and dataset individually and adjust the heatmap limits on line 763 , using the provided method- and dataset-specific min and max predicted values.
CREATE_HEATMAPS = False
DO_TILE_AURC = True
USE_NDWI_INDEX = False
USE_BURNT_INDEX = False
USE_NDVI_INDEX = False
USE_BASELINE = False

assert sum([USE_NDWI_INDEX, USE_BURNT_INDEX, USE_NDVI_INDEX, USE_BASELINE]) in [0, 1]
assert sum([CREATE_HEATMAPS, DO_TILE_AURC]) == 1


class DeeperVAE(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int],
        hidden_channels: List[int],
        latent_dim: int,
        extra_depth_on_scale: int,
        initialized_metrics: List,
        learning_rate: float,
        weight_decay: float,
        margin_size: float,
        variable_margin: bool,
        log_all_metrics: bool,
        dataset_valid,
        **kwargs,
    ) -> None:
        super().__init__()
        assert (
            input_shape[1] >= 2 ** len(hidden_channels)
        ), "Cannot have so many downscaling layers"
        self.latent_dim = latent_dim
        print("\nLATENT SPACE size:", latent_dim)
        self.idxxx = 0

        # Calculate size of encoder output
        encoder_output_width = int(input_shape[1] / (2 ** len(hidden_channels)))
        encoder_output_dim = int(encoder_output_width**2 * hidden_channels[-1])
        self.encoder_output_shape = (
            hidden_channels[-1],
            encoder_output_width,
            encoder_output_width,
        )

        if encoder_output_dim < latent_dim:
            raise UserWarning(
                f"Encoder output dim {encoder_output_dim} is smaller than latent dim {latent_dim}."
                + "This means the bottle neck is tighter than intended."
            )

        in_channels = input_shape[0]
        self.encoder = self._build_encoder(
            [in_channels] + hidden_channels, extra_depth_on_scale
        )
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_output_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, encoder_output_dim)
        self.decoder = self._build_decoder(
            hidden_channels[::-1] + [in_channels], extra_depth_on_scale
        )
        self.backup_metrics = initialized_metrics
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.margin_size = margin_size
        self.variable_margin = variable_margin
        self.log_all_metrics = log_all_metrics
        self.dataset = dataset_valid
        # self.save_hyperparameters("hidden_channels", "extra_depth_on_scale", "learning_rate", "weight_decay", "margin_size", "variable_margin")

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        # for generating the model summary uncomment this
        # return mu
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.encoder_output_shape)
        result = self.decoder(result)
        return result

    def forward(self, input, **kwargs) -> List[Tensor]:
        # forward used for training
        anchor, positive, negative = input
        anchor_mu, anchor_logvar = self.encode(anchor)
        positive_mu, positive_logvar = self.encode(positive)
        negative_mu, negative_logvar = self.encode(negative)

        anchor_z = self.reparameterize(anchor_mu, anchor_logvar)
        positive_z = self.reparameterize(positive_mu, positive_logvar)
        negative_z = self.reparameterize(negative_mu, negative_logvar)

        return [
            [self.decode(anchor_z), anchor_mu, anchor_logvar],
            [self.decode(positive_z), positive_mu, positive_logvar],
            [self.decode(negative_z), negative_mu, negative_logvar],
        ]

    """def forward(self, input):
        #this forward was used for generating the architecture and models the production use
        anchor = input
        anchor_mu = self.encode(anchor)
        return anchor_mu"""

    def _tripletLoss(
        self,
        mu_anchor,
        mu_positive,
        mu_negative,
        change_proportion,
    ):
        distance_pos = 1 - F.cosine_similarity(mu_anchor, mu_positive, dim=-1)
        distance_neg = 1 - F.cosine_similarity(mu_anchor, mu_negative, dim=-1)
        # Triplet loss formulation
        if self.variable_margin:
            triplet_loss = torch.relu(distance_pos - distance_neg + change_proportion)
        else:
            triplet_loss = torch.relu(
                distance_pos - distance_neg + self.margin_size
            )

        return triplet_loss.sum()

    def loss_function(
        self, input: [Tensor, Tensor, Tensor], results: Any, **kwargs
    ) -> Dict:
        """
        Computes the VAE loss function.
        :param args:
        :param kwargs:
        :return:
        """
        # invalid_mask = torch.isnan(input)
        change_proportion = input[3]
        mu_anchor = results[0][1]
        mu_positive = results[1][1]
        mu_negative = results[2][1]

        triplet_loss = self._tripletLoss(
            mu_anchor,
            mu_positive,
            mu_negative,
            change_proportion,
        )
        loss = triplet_loss
        return loss

    def on_train_epoch_start(self) -> None:
        pass
        # self.loss_value = 0
        # self.dataset_len = 0

    def training_step(self, batch, batch_idx):
        anchor, positive, negative, change_proportion = batch
        results = self.forward([anchor, positive, negative])
        loss = self.loss_function(
            [anchor, positive, negative, change_proportion], results
        )
        """with torch.no_grad():
            self.loss_value += loss.detach().cpu().numpy()
            self.dataset_len += anchor.size(0)"""

        return loss

    def on_train_epoch_end(self) -> None:
        pass
        # self.log("mean loss", self.loss_value / self.dataset_len)
        # print(f"mean loss: {self.loss_value/self.dataset_len}")

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.delete_indexes = set()
        self.initialized_metrics = copy.deepcopy(self.backup_metrics)

    def validation_step(self, batch, batch_idx):
        (
            mask,
            before_image,
            after_image,
            event_idx,
            before_picture_idx,
            tile_idx,
            before_nan,
            after_nan,
        ) = batch
        tile_side = mask.shape[-1]
        tile_size = pow(tile_side, 2)
        cloud_pixels_after = torch.sum(after_image[:, 10, :, :])
        cloud_pixels_before = torch.sum(before_image[:, 10, :, :])
        before_mu, before_logvar = self.encode(before_image[:, :10, :, :])
        # np.save(f"exports/embeddings/{self.counter}.npy", before_mu)
        # self.counter += 1
        after_mu, after_logvar = self.encode(after_image[:, :10, :, :])
        distances = 1 - F.cosine_similarity(before_mu, after_mu, dim=-1)
        distance = float(distances.squeeze(0))

        if USE_NDWI_INDEX or USE_BURNT_INDEX or USE_NDVI_INDEX:
            before_image = unnormalize_tile(before_image.cpu())
            after_image = unnormalize_tile(after_image.cpu())
            if USE_NDWI_INDEX:
                b3_before = before_image[:, 1, :, :]
                b8_before = before_image[:, 6, :, :]
                ndwi_before = ((b3_before - b8_before) / (b3_before + b8_before)) + 1
                b3_after = after_image[:, 1, :, :]
                b8_after = after_image[:, 6, :, :]
                ndwi_after = ((b3_after - b8_after) / (b3_after + b8_after)) + 1
                distance = float(ndwi_after.mean() - ndwi_before.mean())
            elif USE_BURNT_INDEX:
                b8_before = before_image[:, 6, :, :]
                b12_before = before_image[:, 9, :, :]
                # burnt index is reverse, low value means burnt area
                burnt_before = abs(
                    ((b8_before - b12_before) / (b8_before + b12_before)) + 1 - 2
                )
                b8_after = after_image[:, 6, :, :]
                b12_after = after_image[:, 9, :, :]
                burnt_after = abs(
                    ((b8_after - b12_after) / (b8_after + b12_after)) + 1 - 2
                )
                distance = float(burnt_after.mean() - burnt_before.mean())
            else:
                b4_before = before_image[:, 2, :, :]
                b8_before = before_image[:, 6, :, :]
                ndvi_before = ((b8_before - b4_before) / (b8_before + b4_before)) + 1
                b4_after = after_image[:, 2, :, :]
                b8_after = after_image[:, 6, :, :]
                ndvi_after = ((b8_after - b4_after) / (b8_after + b4_after)) + 1
                # NDVI index measure vegation, in case of disaster there is less vegetation after
                distance = float(ndvi_before.mean() - ndvi_after.mean())

        if USE_BASELINE:
            distances = 1 - F.cosine_similarity(
                before_image.flatten(), after_image.flatten(), dim=0
            )
            distance = float(distances)

        # real_portion = float(torch.sum(mask) / torch.numel(mask))
        real_changed_pixels = torch.sum(mask)

        event_idx = int(event_idx.squeeze(0))
        tile_idx = int(tile_idx.squeeze(0))
        before_picture_idx = int(before_picture_idx.squeeze(0))

        if cloud_pixels_after / tile_size >= 0.25 or after_nan:
            self.delete_indexes.add((event_idx, tile_idx))
            return mask
        if cloud_pixels_before / tile_size >= 0.25 or before_nan:
            self.delete_indexes.add((event_idx, tile_idx, 0, before_picture_idx))
            return mask

        self.initialized_metrics[event_idx][tile_idx][0][
            before_picture_idx
        ] = distance
        self.initialized_metrics[event_idx][tile_idx][1] = real_changed_pixels

        return mask

    def on_validation_epoch_end(self):
        self.delete_indexes = list(self.delete_indexes)
        self.delete_indexes.sort(reverse=True)

        if CREATE_HEATMAPS:
            self.create_heatmaps()
            return

        for delete_list in self.delete_indexes:
            if len(delete_list) == 2:
                del self.initialized_metrics[delete_list[0]][delete_list[1]]
            else:
                del self.initialized_metrics[delete_list[0]][delete_list[1]][
                    delete_list[2]
                ][delete_list[3]]

        one_day_metrics = []
        avg_day_metrics = []
        min_day_metrics = []

        for event_list in self.initialized_metrics:
            one_day_event_metrics = []
            avg_event_metrics = []
            min_event_metrics = []
            for tile_list in event_list:
                if tile_list[0]:
                    tile_one_day_metrics = (tile_list[0][-1], tile_list[1])
                    tile_avg_metrics = (mean(tile_list[0]), tile_list[1])
                    tile_min_metrics = (min(tile_list[0]), tile_list[1])
                    one_day_event_metrics.append(tile_one_day_metrics)
                    avg_event_metrics.append(tile_avg_metrics)
                    min_event_metrics.append(tile_min_metrics)
            one_day_metrics.append(one_day_event_metrics)
            avg_day_metrics.append(avg_event_metrics)
            min_day_metrics.append(min_event_metrics)

        concatanated_one_day_metrics = []
        concatanated_avg_day_metrics = []
        concatanated_min_day_metrics = []

        for i in range(len(one_day_metrics)):
            concatanated_one_day_metrics += one_day_metrics[i]
            concatanated_avg_day_metrics += avg_day_metrics[i]
            concatanated_min_day_metrics += min_day_metrics[i]

        # New metrics computation as per reviewer's request
        self.compute_new_metrics(
            concatanated_one_day_metrics,
            concatanated_avg_day_metrics,
            concatanated_min_day_metrics,
        )

        portion = 0.0
        recalled_portions = []
        for i in range(100):
            portion += 0.01
            recalled_portions.append(round(portion, 2))

        self.avg_list = []
        self.min_list = []
        self.one_list = []
        if DO_TILE_AURC:
            for pixel_num in recalled_portions:
                self.compute_metrics(
                    concatanated_one_day_metrics,
                    concatanated_avg_day_metrics,
                    concatanated_min_day_metrics,
                    pixel_num,
                    only_auc=True,
                )
            self.print_and_log_info(
                "area_under_the_curve_avg",
                auc(recalled_portions, self.avg_list),
                avg=False,
            )
            self.print_and_log_info(
                "area_under_the_curve_min",
                auc(recalled_portions, self.min_list),
                avg=False,
            )
            self.print_and_log_info(
                "area_under_the_curve_one",
                auc(recalled_portions, self.one_list),
                avg=False,
            )
            return
        else:
            whole_statistics = self.compute_metrics(
                concatanated_one_day_metrics,
                concatanated_avg_day_metrics,
                concatanated_min_day_metrics,
                1,
            )

            mean_statistics = self.get_list_of_metrics(
                one_day_metrics, avg_day_metrics, min_day_metrics
            )

            names = [
                "one_memory_precision_recall_pixel",
                "all_avg_memory_precision_recall_pixel",
                "all_min_memory_precision_recall_pixel",
                "one_memory_precision_recall_tile",
                "all_avg_memory_precision_recall_tile",
                "all_min_memory_precision_recall_tile",
                "one_memory_corr_whole",
                "all_avg_memory_corr_whole",
                "all_min_memory_corr_whole",
                "one_memory_corr_retrieved",
                "all_avg_memory_corr_retrieved",
                "all_min_memory_corr_retrieved",
            ]

            whole_names = [f"whole_{name}" for name in names]

            for name, statistic in zip(names, mean_statistics):
                if self.log_all_metrics:
                    self.print_and_log_info(name, statistic, avg=True)

            self.print_and_log_info(
                "whole_average", [whole_statistics[i] for i in [1, 4, 10]], avg=True
            )
            for name, statistic in zip(whole_names, whole_statistics):
                self.print_and_log_info(name, statistic, avg=False)

            return

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: Mean of the latent Gaussian [B x D]
        :param logvar: Standard deviation of the latent Gaussian [B x D]
        :return: [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def print_and_log_info(self, name_string, statistic, avg=True):
        if avg:
            statistic = mean(statistic)
            name_string = f"{name_string}_avg"
        else:
            statistic = float(statistic)
            name_string = f"{name_string}_overall"
        print(f"{name_string}: {statistic}")
        self.log(
            name_string,
            statistic,
            prog_bar=False,
        )

    def compute_metrics(
        self, event_one, event_avg, event_min, recalled_portion, only_auc=False
    ):
        changed_only = [f for f in event_one if f[1] > 0]
        stop_index = int(len(changed_only) * recalled_portion)
        should_have_been_recalled = sorted(event_one, key=lambda x: x[1], reverse=True)[
            :stop_index
        ]

        if only_auc:
            one_memory_recalled = sorted(event_one, key=lambda x: x[0], reverse=True)[
                :stop_index
            ]
            avg_recalled = sorted(event_avg, key=lambda x: x[0], reverse=True)[
                :stop_index
            ]
            min_recalled = sorted(event_min, key=lambda x: x[0], reverse=True)[
                :stop_index
            ]

            num_of_true_changed_pixels = sum([f[1] for f in should_have_been_recalled])
            num_of_true_changed_pixels = (
                num_of_true_changed_pixels if num_of_true_changed_pixels > 0 else 1
            )
            num_of_recalled_pixels_avg = sum([f[1] for f in avg_recalled])
            num_of_recalled_pixels_min = sum([f[1] for f in min_recalled])
            num_of_recalled_pixels_one_memory = sum([f[1] for f in one_memory_recalled])

            avg_memory_precision_recall = float(
                num_of_recalled_pixels_avg / num_of_true_changed_pixels
            )
            min_memory_precision_recall = float(
                num_of_recalled_pixels_min / num_of_true_changed_pixels
            )
            one_memory_precision_recall = float(
                num_of_recalled_pixels_one_memory / num_of_true_changed_pixels
            )

            self.avg_list.append(avg_memory_precision_recall)
            self.min_list.append(min_memory_precision_recall)
            self.one_list.append(one_memory_precision_recall)

            if recalled_portion == 0.05:
                minimal_changed_count = len([f for f in event_one if f[1] > 5])
                all_count = len(event_one)
                one_memory_sorted = sorted(
                    event_one, key=lambda x: x[0], reverse=True
                )
                avg_recalled = sorted(event_avg, key=lambda x: x[0], reverse=True)
                min_recalled = sorted(event_min, key=lambda x: x[0], reverse=True)

                one_memory_last_index = 0
                avg_memory_last_index = 0
                min_memory_last_index = 0
                counter = 0

                for one, avg, min in zip(one_memory_sorted, avg_recalled, min_recalled):
                    if one[1] > 51:
                        one_memory_last_index = counter
                    if avg[1] > 51:
                        avg_memory_last_index = counter
                    if min[1] > 51:
                        min_memory_last_index = counter
                    counter += 1

                self.print_and_log_info(
                    "percentage_of_changed", minimal_changed_count / all_count, avg=False
                )
                self.print_and_log_info("Total N of tiles", len(event_one), avg=False)
                self.print_and_log_info(
                    "downlink_amount_one", one_memory_last_index / all_count, avg=False
                )
                self.print_and_log_info(
                    "downlink_amount_avg", avg_memory_last_index / all_count, avg=False
                )
                self.print_and_log_info(
                    "downlink_amount_min", min_memory_last_index / all_count, avg=False
                )
            return

        minimal_changed_pixels = should_have_been_recalled[stop_index - 1][1]

        one_memory_recalled = sorted(event_one, key=lambda x: x[0], reverse=True)[
            :stop_index
        ]
        avg_recalled = sorted(event_avg, key=lambda x: x[0], reverse=True)[:stop_index]
        min_recalled = sorted(event_min, key=lambda x: x[0], reverse=True)[:stop_index]

        num_of_true_changed_pixels = sum([f[1] for f in should_have_been_recalled])
        num_of_true_changed_pixels = (
            num_of_true_changed_pixels if num_of_true_changed_pixels > 0 else 1
        )
        num_of_true_changed_tiles = stop_index
        num_of_true_changed_tiles = stop_index if stop_index > 0 else 1

        num_of_recalled_pixels_one_memory = sum([f[1] for f in one_memory_recalled])
        num_of_recalled_tiles_one_memory = len(
            [f[1] for f in one_memory_recalled if f[1] > minimal_changed_pixels]
        )
        num_of_recalled_pixels_avg = sum([f[1] for f in avg_recalled])
        num_of_recalled_tiles_avg = len(
            [f[1] for f in avg_recalled if f[1] > minimal_changed_pixels]
        )
        num_of_recalled_pixels_min = sum([f[1] for f in min_recalled])
        num_of_recalled_tiles_min = len(
            [f[1] for f in min_recalled if f[1] > minimal_changed_pixels]
        )

        one_memory_precision_recall = float(
            num_of_recalled_pixels_one_memory / num_of_true_changed_pixels
        )
        avg_memory_precision_recall = float(
            num_of_recalled_pixels_avg / num_of_true_changed_pixels
        )
        min_memory_precision_recall = float(
            num_of_recalled_pixels_min / num_of_true_changed_pixels
        )
        self.avg_list.append(avg_memory_precision_recall)
        self.min_list.append(min_memory_precision_recall)
        self.one_list.append(one_memory_precision_recall)

        one_memory_precision_recall_tile = float(
            num_of_recalled_tiles_one_memory / num_of_true_changed_tiles
        )
        avg_memory_precision_recall_tile = float(
            num_of_recalled_tiles_avg / num_of_true_changed_tiles
        )
        min_memory_precision_recall_tile = float(
            num_of_recalled_tiles_min / num_of_true_changed_tiles
        )

        one_memory_corr_whole = spearmanr(
            [x[0] for x in event_one], [x[1] for x in event_one]
        ).statistic
        avg_memory_corr_whole = spearmanr(
            [x[0] for x in event_avg], [x[1] for x in event_avg]
        ).statistic
        min_memory_corr_whole = spearmanr(
            [x[0] for x in event_min], [x[1] for x in event_min]
        ).statistic
        one_memory_corr_retrieved = spearmanr(
            [x[0] for x in one_memory_recalled], [x[1] for x in one_memory_recalled]
        ).statistic
        avg_memory_corr_retrieved = spearmanr(
            [x[0] for x in avg_recalled], [x[1] for x in avg_recalled]
        ).statistic
        min_memory_corr_retrieved = spearmanr(
            [x[0] for x in min_recalled], [x[1] for x in min_recalled]
        ).statistic

        return (
            one_memory_precision_recall,
            avg_memory_precision_recall,
            min_memory_precision_recall,
            one_memory_precision_recall_tile,
            avg_memory_precision_recall_tile,
            min_memory_precision_recall_tile,
            one_memory_corr_whole,
            avg_memory_corr_whole,
            min_memory_corr_whole,
            one_memory_corr_retrieved,
            avg_memory_corr_retrieved,
            min_memory_corr_retrieved,
        )

    def get_list_of_metrics(self, one_day_metrics, avg_day_metrics, min_day_metrics):
        event_id = 0
        one_memory_precision_recall_pixel_list = []
        avg_memory_precision_recall_pixel_list = []
        min_memory_precision_recall_pixel_list = []
        one_memory_precision_recall_list = []
        avg_memory_precision_recall_list = []
        min_memory_precision_recall_list = []
        one_memory_corr_whole_list = []
        avg_memory_corr_whole_list = []
        min_memory_corr_whole_list = []
        one_memory_corr_retrieved_list = []
        avg_memory_corr_retrieved_list = []
        min_memory_corr_retrieved_list = []

        for event_one, event_avg, event_min in zip(
            one_day_metrics, avg_day_metrics, min_day_metrics
        ):
            (
                one_memory_precision_recall,
                avg_memory_precision_recall,
                min_memory_precision_recall,
                one_memory_precision_recall_tile,
                avg_memory_precision_recall_tile,
                min_memory_precision_recall_tile,
                one_memory_corr_whole,
                avg_memory_corr_whole,
                min_memory_corr_whole,
                one_memory_corr_retrieved,
                avg_memory_corr_retrieved,
                min_memory_corr_retrieved,
            ) = self.compute_metrics(event_one, event_avg, event_min, 1)

            one_memory_precision_recall_pixel_list.append(one_memory_precision_recall)
            avg_memory_precision_recall_pixel_list.append(avg_memory_precision_recall)
            min_memory_precision_recall_pixel_list.append(min_memory_precision_recall)
            one_memory_precision_recall_list.append(one_memory_precision_recall_tile)
            avg_memory_precision_recall_list.append(avg_memory_precision_recall_tile)
            min_memory_precision_recall_list.append(min_memory_precision_recall_tile)
            one_memory_corr_whole_list.append(one_memory_corr_whole)
            avg_memory_corr_whole_list.append(avg_memory_corr_whole)
            min_memory_corr_whole_list.append(min_memory_corr_whole)
            one_memory_corr_retrieved_list.append(one_memory_corr_retrieved)
            avg_memory_corr_retrieved_list.append(avg_memory_corr_retrieved)
            min_memory_corr_retrieved_list.append(min_memory_corr_retrieved)

            if self.log_all_metrics:
                event_metrics = {
                    f"{event_id}_event": event_id,
                    f"{event_id}_one_memory_whole_corr": one_memory_corr_whole,
                    f"{event_id}_avg_memory_whole_corr": avg_memory_corr_whole,
                    f"{event_id}_min_memory_whole_corr": min_memory_corr_whole,
                    f"{event_id}_one_memory_recalled_corr": one_memory_corr_retrieved,
                    f"{event_id}_avg_memory_recalled_corr": avg_memory_corr_retrieved,
                    f"{event_id}_min_memory_recalled_corr": min_memory_corr_retrieved,
                    f"{event_id}_one_memory_precision_recall": one_memory_precision_recall,
                    f"{event_id}_avg_memory_precision_recall": avg_memory_precision_recall,
                    f"{event_id}_min_memory_precision_recall": min_memory_precision_recall,
                }
                for k, v in event_metrics.items():
                    self.log(k, v, prog_bar=False)
            event_id += 1
        return [
            one_memory_precision_recall_pixel_list,
            avg_memory_precision_recall_pixel_list,
            min_memory_precision_recall_pixel_list,
            one_memory_precision_recall_list,
            avg_memory_precision_recall_list,
            min_memory_precision_recall_list,
            one_memory_corr_whole_list,
            avg_memory_corr_whole_list,
            min_memory_corr_whole_list,
            one_memory_corr_retrieved_list,
            avg_memory_corr_retrieved_list,
            min_memory_corr_retrieved_list,
        ]

    def create_heatmaps(self):
        # [event][tile_id][0][before_picture_id]
        heatmap_metrics = self.initialized_metrics
        for delete_list in self.delete_indexes:
            if len(delete_list) == 2:
                new_list = [np.nan]
                heatmap_metrics[delete_list[0]][delete_list[1]][0] = new_list
            else:
                if (
                    len(
                        heatmap_metrics[delete_list[0]][delete_list[1]][
                            delete_list[2]
                        ]
                    )
                    > 1
                ):
                    del heatmap_metrics[delete_list[0]][delete_list[1]][
                        delete_list[2]
                    ][delete_list[3]]
                else:
                    heatmap_metrics[delete_list[0]][delete_list[1]][delete_list[2]][
                        delete_list[3]
                    ] = np.nan

        for event_id, event in enumerate(heatmap_metrics):
            mask = self.dataset.change_masks[event_id]
            avg_event_heatmap = np.zeros((mask.shape[0], mask.shape[1]))
            min_event_heatmap = np.zeros((mask.shape[0], mask.shape[1]))
            one_event_heatmap = np.zeros((mask.shape[0], mask.shape[1]))
            for tile_id, tile in enumerate(event):
                tile_height_index, tile_width_index = (
                    tiling.get_tile_height_and_width_indexes(mask, tile_id, 32)
                )

                cos_distances = tile[0]

                avg_event_heatmap_part = mean(cos_distances)
                min_event_heatmap_part = min(cos_distances)
                one_event_heatmap_part = cos_distances[-1]

                avg_event_heatmap_part_mask = (
                    avg_event_heatmap[
                        tile_height_index : tile_height_index + 32,
                        tile_width_index : tile_width_index + 32,
                    ]
                    == 0
                )
                min_event_heatmap_part_mask = (
                    min_event_heatmap[
                        tile_height_index : tile_height_index + 32,
                        tile_width_index : tile_width_index + 32,
                    ]
                    == 0
                )
                one_event_heatmap_part_mask = (
                    one_event_heatmap[
                        tile_height_index : tile_height_index + 32,
                        tile_width_index : tile_width_index + 32,
                    ]
                    == 0
                )

                avg_event_heatmap[
                    tile_height_index : tile_height_index + 32,
                    tile_width_index : tile_width_index + 32,
                ][avg_event_heatmap_part_mask] = avg_event_heatmap_part
                min_event_heatmap[
                    tile_height_index : tile_height_index + 32,
                    tile_width_index : tile_width_index + 32,
                ][min_event_heatmap_part_mask] = min_event_heatmap_part
                one_event_heatmap[
                    tile_height_index : tile_height_index + 32,
                    tile_width_index : tile_width_index + 32,
                ][one_event_heatmap_part_mask] = one_event_heatmap_part

            self.save_heatmap_with_colorbar(
                avg_event_heatmap, f"exports/{event_id}_avg.png"
            )
            self.save_heatmap_with_colorbar(
                min_event_heatmap, f"exports/{event_id}_min.png"
            )
            self.save_heatmap_with_colorbar(
                one_event_heatmap, f"exports/{event_id}_one.png"
            )

    def save_heatmap_with_colorbar(self, data, filename):
        nan_color = (1, 1, 1)  # RGB value for white
        # Create a copy of the 'viridis' colormap and set nan values to white
        # limits are rounded on 2 decimals
        # limits for my test dataset (floods), avg, baseline - (0,1.62), ndwi - (-0.37, 0.75), my_small - (0,1.64), ravaen_medium - (0,1.5)
        # limits for ravaen landslides, min, baseline - (0,1.46), NDVI - (-0.87,0.6), my_large - (0.01, 1.12), ravaen_medium - (0.01,1.05)
        # limits for ravaen floods, min, baseline - (0,1.87), ndwi - (-0.74,0.79), ravaen_small - (0,1.29), my_small - (0,1.56)
        # limits for ravaen hurricanes, min, baseline - (0,1.69), ndvi - (-0.33,0.95), ravaen_small - (0,1.17), my_big - (0,1.51)
        # limits for ravaen fires, min NBR - (-0.73, 1.07), baseline - (0, 1.56), ravaen_medium - (0,1.31), my_big (0,0.94)
        limits = (-0.87, 0.6)
        cmap = plt.cm.get_cmap("viridis").copy()
        cmap.set_bad(color=nan_color)
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1])
        # Adjust cmap as needed
        ax.set_axis_off()
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.02,
                ax.get_position().height,
            ]
        )
        plt.colorbar(im, cax=cax)
        cax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        cax.yaxis.set_tick_params(labelsize=18)
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    @staticmethod
    def _build_encoder(channels, extra_depth):
        in_channels = channels[0]
        encoder = []
        for out_channels in channels[1:]:
            encoder += [
                DownConv(
                    in_channels, out_channels, activation=nn.LeakyReLU, batchnorm=True
                )
            ]
            if extra_depth > 0:
                encoder += [
                    ResConvBlock(
                        out_channels,
                        out_channels,
                        activation=nn.LeakyReLU,
                        batchnorm=True,
                        depth=extra_depth,
                    )
                ]
            # for next time round loop
            in_channels = out_channels
        return nn.Sequential(*encoder)

    @staticmethod
    def _build_decoder(channels, extra_depth):
        in_channels = channels[0]
        decoder = []
        up_activation = nn.LeakyReLU
        res_activation = nn.LeakyReLU
        for i, out_channels in enumerate(channels[1:]):
            # if last layer use linear activation
            is_last_layer = i == (len(channels) - 2)
            if is_last_layer:
                up_activation = None
            decoder += [
                UpConv(
                    in_channels,
                    out_channels,
                    upsample_method="nearest",
                    activation=up_activation,
                    batchnorm=not is_last_layer,
                )
            ]
            if extra_depth > 0:
                decoder += [
                    ResConvBlock(
                        out_channels,
                        out_channels,
                        activation=res_activation,
                        batchnorm=not is_last_layer,
                        depth=extra_depth,
                    )
                ]
            # for next time round loop
            in_channels = out_channels
        return nn.Sequential(*decoder)
    
    def compute_new_metrics(self, one_day_metrics, avg_day_metrics, min_day_metrics, threshold=51):
        """
        Computes standard metrics at the tile level.
        A tile is considered "changed" if the number of changed pixels is above a certain threshold.
        """
        print("\n--- Computing New Metrics (Tile Level) ---")
    
        # Define a tile as "changed" if its change pixel count is > threshold
        y_true_one = [1 if tile[1] > threshold else 0 for tile in one_day_metrics]
        y_scores_one = [tile[0] for tile in one_day_metrics]
    
        y_true_avg = [1 if tile[1] > threshold else 0 for tile in avg_day_metrics]
        y_scores_avg = [tile[0] for tile in avg_day_metrics]
    
        y_true_min = [1 if tile[1] > threshold else 0 for tile in min_day_metrics]
        y_scores_min = [tile[0] for tile in min_day_metrics]
    
        # Calculate optimal threshold for F1-Score
        def find_optimal_threshold(y_true, y_scores):
            best_f1 = 0
            best_thresh = 0
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
            for precision, recall, threshold in zip(precisions, recalls, thresholds):
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = threshold
            return best_thresh
    
        optimal_thresh_one = find_optimal_threshold(y_true_one, y_scores_one)
        optimal_thresh_avg = find_optimal_threshold(y_true_avg, y_scores_avg)
        optimal_thresh_min = find_optimal_threshold(y_true_min, y_scores_min)
    
        # Binarize predictions using optimal threshold
        y_pred_one = [1 if score >= optimal_thresh_one else 0 for score in y_scores_one]
        y_pred_avg = [1 if score >= optimal_thresh_avg else 0 for score in y_scores_avg]
        y_pred_min = [1 if score >= optimal_thresh_min else 0 for score in y_scores_min]
    
        # F1-Score, Precision, and Recall
        f1_one = f1_score(y_true_one, y_pred_one)
        precision_one = precision_score(y_true_one, y_pred_one)
        recall_one = recall_score(y_true_one, y_pred_one)
    
        f1_avg = f1_score(y_true_avg, y_pred_avg)
        precision_avg = precision_score(y_true_avg, y_pred_avg)
        recall_avg = recall_score(y_true_avg, y_pred_avg)
    
        f1_min = f1_score(y_true_min, y_pred_min)
        precision_min = precision_score(y_true_min, y_pred_min)
        recall_min = recall_score(y_true_min, y_pred_min)
    
        self.print_and_log_info("Tile-level F1-Score (one memory)", f1_one, avg=False)
        self.print_and_log_info("Tile-level Precision (one memory)", precision_one, avg=False)
        self.print_and_log_info("Tile-level Recall (one memory)", recall_one, avg=False)
    
        self.print_and_log_info("Tile-level F1-Score (avg memory)", f1_avg, avg=False)
        self.print_and_log_info("Tile-level Precision (avg memory)", precision_avg, avg=False)
        self.print_and_log_info("Tile-level Recall (avg memory)", recall_avg, avg=False)
    
        self.print_and_log_info("Tile-level F1-Score (min memory)", f1_min, avg=False)
        self.print_and_log_info("Tile-level Precision (min memory)", precision_min, avg=False)
        self.print_and_log_info("Tile-level Recall (min memory)", recall_min, avg=False)
    
        # AUPRC (Area Under the Precision-Recall Curve)
        auprc_one = average_precision_score(y_true_one, y_scores_one)
        auprc_avg = average_precision_score(y_true_avg, y_scores_avg)
        auprc_min = average_precision_score(y_true_min, y_scores_min)
    
        self.print_and_log_info("Tile-level AUPRC (one memory)", auprc_one, avg=False)
        self.print_and_log_info("Tile-level AUPRC (avg memory)", auprc_avg, avg=False)
        self.print_and_log_info("Tile-level AUPRC (min memory)", auprc_min, avg=False)
    
        # Precision@K and Mean Average Precision (MAP) for ranking quality
        def precision_at_k(y_true, y_scores, k):
            sorted_indices = np.argsort(y_scores)[::-1]
            top_k_true = np.array(y_true)[sorted_indices[:k]]
            return np.mean(top_k_true) if k > 0 else 0
    
        def average_precision(y_true, y_scores):
            y_true_sorted = [y for _, y in sorted(zip(y_scores, y_true), reverse=True)]
            changed_indices = [i for i, label in enumerate(y_true_sorted) if label == 1]
            if not changed_indices:
                return 0
            
            ap_sum = 0
            for k in range(1, len(y_true_sorted) + 1):
                if y_true_sorted[k-1] == 1:
                    ap_sum += precision_at_k(y_true_sorted, y_true_sorted, k)
            
            return ap_sum / len(changed_indices)
    
        # Since we have a single set of scores, MAP is equivalent to Average Precision
        map_one = average_precision(y_true_one, y_scores_one)
        map_avg = average_precision(y_true_avg, y_scores_avg)
        map_min = average_precision(y_true_min, y_scores_min)
    
        self.print_and_log_info("Tile-level MAP (one memory)", map_one, avg=False)
        self.print_and_log_info("Tile-level MAP (avg memory)", map_avg, avg=False)
        self.print_and_log_info("Tile-level MAP (min memory)", map_min, avg=False)
    
        # Precision@K for a few K values
        k_values = [10, 50, 100]
        for k in k_values:
            p_at_k_one = precision_at_k(y_true_one, y_scores_one, k)
            p_at_k_avg = precision_at_k(y_true_avg, y_scores_avg, k)
            p_at_k_min = precision_at_k(y_true_min, y_scores_min, k)
    
            self.print_and_log_info(f"Tile-level P@{k} (one memory)", p_at_k_one, avg=False)
            self.print_and_log_info(f"Tile-level P@{k} (avg memory)", p_at_k_avg, avg=False)
            self.print_and_log_info(f"Tile-level P@{k} (min memory)", p_at_k_min, avg=False)


class ConvBlock(nn.Module):
    """
    Convolutional block which preserves the height and width of the input image.
    (convolution => [BN] => LeakyReLU) * depth
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        activation=nn.LeakyReLU,
        batchnorm=True,
    ):
        super().__init__()
        layers = []
        for n in range(1, depth + 1):
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            if batchnorm:
                layers += [nn.BatchNorm2d(out_channels)]
            if activation is not None:
                layers += [activation()]
            in_channels = out_channels
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class ResConvBlock(ConvBlock):
    def forward(self, x):
        dx = self.conv_block(x)
        return x + dx


class DownConv(nn.Module):
    """Downscaling block"""

    def __init__(
        self, in_channels, out_channels, activation=nn.LeakyReLU, batchnorm=True
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        ]
        if batchnorm:
            layers += [nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers += [activation()]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upscaling layer with single convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        upsample_method="nearest",
        activation=nn.LeakyReLU,
        batchnorm=True,
    ):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if upsample_method in ["nearest", "linear", "bilinear", "bicubic"]:
            align_corners = None if upsample_method == "nearest" else True
            self.up = nn.Sequential(
                nn.Upsample(
                    scale_factor=2, mode=upsample_method, align_corners=align_corners
                ),
                ConvBlock(
                    in_channels,
                    out_channels,
                    depth=1,
                    activation=activation,
                    batchnorm=batchnorm,
                ),
            )
        # add the single convolution above to keep things even between the number of convolutions
        # in upsamplking and downsampling, regardless of upsampling method
        elif upsample_method == "transpose":
            layers = [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
            if batchnorm:
                layers += [nn.BatchNorm2d(out_channels)]
            if activation is not None:
                layers += [activation()]
            self.up = nn.Sequential(*layers)
        else:
            raise NotImplementedError(
                f"Upsample method has not been implemented: {upsample_method}"
            )

    def forward(self, x):
        return self.up(x)