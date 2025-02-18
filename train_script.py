import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datas
import DeeperSiameseVae
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from load_validation_data import get_validation_data
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import wandb
from save_tiles_with_npy_mask import tile_original_images
from preprocess_train_tiles import get_anchor_positive_negative_tiles

DEVICE = "cpu"


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")


def initialize_model(
    initialized_metrics,
    size,
    learning_rate,
    weight_decay,
    margin_size,
    variable_margin,
    log_all_metrics,
    dataset,
):
    latent_dim = 128
    input_shape = (10, 32, 32)
    if size == "large":
        path = "./models/ravaen_modified_ckpt/large_modified_checkpoint.ckpt"
        hidden_channels = [32, 64, 128]
        extra_depth = 2
    if size == "medium":
        path = "./models/ravaen_modified_ckpt/medium_modified_checkpoint.ckpt"
        hidden_channels = [32, 64, 128]
        extra_depth = 0
    if size == "small":
        path = "./models/ravaen_modified_ckpt/small_modified_checkpoint.ckpt"
        hidden_channels = [16, 32, 64]
        extra_depth = 0
    model = DeeperSiameseVae.DeeperVAE.load_from_checkpoint(
        path,
        map_location="cpu",
        hidden_channels=hidden_channels,
        extra_depth_on_scale=extra_depth,
        latent_dim=latent_dim,
        input_shape=input_shape,
        initialized_metrics=initialized_metrics,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        margin_size=margin_size,
        variable_margin=variable_margin,
        log_all_metrics=log_all_metrics,
        dataset_valid=dataset,
    )
    model.to(DEVICE)
    return model


def train_model(
    model_size,
    learning_rate,
    weight_decay,
    margin_size,
    variable_margin,
    batch_size,
    log_all_metrics,
    n_epochs,
    before_images,
    after_images,
    change_masks,
    final_tiles_path,
    use_wandb=True,
):
    validation_dataset = datas.ValidTestDataset(
        before_images, after_images, change_masks
    )
    initialized_metrics = validation_dataset.get_initialized_metrics()
    train_dataset = datas.TripletDataset(dataset_path=final_tiles_path)
    model = initialize_model(
        initialized_metrics=initialized_metrics,
        size=model_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        margin_size=margin_size,
        variable_margin=variable_margin,
        log_all_metrics=log_all_metrics,
        dataset=validation_dataset,
    )
    n_cpu = os.cpu_count()
    # Create a dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(n_cpu / 2),
        worker_init_fn=set_worker_sharing_strategy,
    )
    valid_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        worker_init_fn=set_worker_sharing_strategy,
    )
    callbacks = [
        ModelCheckpoint(
            save_top_k=3,
            mode="max",
            monitor="area_under_the_curve_avg_overall",
            filename="checkpoints_train/epoch_{epoch:02d}-step_{step}",
            auto_insert_metric_name=False,
        )
    ]
    if use_wandb:
        logger = WandbLogger(log_model=True)
    else:
        logger = CSVLogger("logs", name="my_exp_name")
    # Create a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=n_epochs,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    model_size = "medium"
    learning_rate = 0.000002
    weight_decay = 0.00001
    margin_size = (0.0, 1)
    stride = 32
    variable_margin = True
    batch_size = 32
    log_all_metrics = False
    n_epochs = 1
    final_tiles_path = (
        f"data/tiled/final_{margin_size[0]}_{margin_size[1]}_{stride}"
    )
    before_images, after_images, change_masks = get_validation_data(
        ravaen=True, disaster_type="floods"
    )
    if not os.path.exists(final_tiles_path):
        tile_original_images(margin_size, stride)
        get_anchor_positive_negative_tiles(final_tiles_path)

    train_model(
        model_size=model_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        margin_size=margin_size,
        variable_margin=variable_margin,
        batch_size=batch_size,
        log_all_metrics=log_all_metrics,
        n_epochs=n_epochs,
        before_images=before_images,
        after_images=after_images,
        change_masks=change_masks,
        final_tiles_path=final_tiles_path,
        use_wandb=False,
    )
