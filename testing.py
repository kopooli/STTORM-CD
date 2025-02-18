import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datas
import DeeperSiameseVae

# import torch
# from torchsummary import summary
from load_validation_data import get_validation_data

DEVICE = "cpu"


def load_model(checkpoint_path, size, initialized_metrics, dataset):
    latent_dim = 128
    input_shape = (10, 32, 32)
    if size == "large":
        hidden_channels = [32, 64, 128]
        extra_depth = 2
    if size == "medium":
        hidden_channels = [32, 64, 128]
        extra_depth = 0
    if size == "small":
        hidden_channels = [16, 32, 64]
        extra_depth = 0
    model = DeeperSiameseVae.DeeperVAE.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        hidden_channels=hidden_channels,
        extra_depth_on_scale=extra_depth,
        latent_dim=latent_dim,
        input_shape=input_shape,
        initialized_metrics=initialized_metrics,
        learning_rate=0.001,
        weight_decay=0.001,
        margin_size=0.5,
        variable_margin=True,
        log_all_metrics=False,
        dataset_valid=dataset,
    )
    model.to(DEVICE)
    return model


if __name__ == "__main__":
    my_checkpoint_path = "models/sttorm_cd_lightning_ckpt/"
    original_path = "models/ravaen_modified_ckpt/"
    # here select the disaster type and dataset
    # disaster types: fires floods hurricanes landslides
    # raven=False uses my floods test dataset
    before_images, after_images, change_masks = get_validation_data(
        ravaen=False, disaster_type="floods"
    )

    test_dataset = datas.ValidTestDataset(before_images, after_images, change_masks)
    initialized_metrics = test_dataset.get_initialized_metrics()
    my_small_model = load_model(
        os.path.join(my_checkpoint_path, "small.ckpt"),
        "small",
        initialized_metrics,
        test_dataset,
    )
    my_medium_model = load_model(
        os.path.join(my_checkpoint_path, "medium.ckpt"),
        "medium",
        initialized_metrics,
        test_dataset,
    )
    my_large_model = load_model(
        os.path.join(my_checkpoint_path, "large.ckpt"),
        "large",
        initialized_metrics,
        test_dataset,
    )
    original_small_model = load_model(
        os.path.join(original_path, "small_modified_checkpoint.ckpt"),
        "small",
        initialized_metrics,
        test_dataset,
    )
    original_medium_model = load_model(
        os.path.join(original_path, "medium_modified_checkpoint.ckpt"),
        "medium",
        initialized_metrics,
        test_dataset,
    )
    original_large_model = load_model(
        os.path.join(original_path, "large_modified_checkpoint.ckpt"),
        "large",
        initialized_metrics,
        test_dataset,
    )
    """input_data = torch.zeros((16,10,32,32))
    summary(original_small_model, input_data)
    summary(original_medium_model, input_data)
    summary(original_large_model, input_data)"""
    n_cpu = os.cpu_count()
    # Create a dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(n_cpu / 2),
        persistent_workers=True,
    )
    # Create a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=1,
    )
    print("original_small")
    trainer.test(model=original_small_model, dataloaders=test_dataloader)
    print("original_medium")
    trainer.test(model=original_medium_model, dataloaders=test_dataloader)
    print("original_large")
    trainer.test(model=original_large_model, dataloaders=test_dataloader)
    print("my_small")
    trainer.test(model=my_small_model, dataloaders=test_dataloader)
    print("my_medium")
    trainer.test(model=my_medium_model, dataloaders=test_dataloader)
    print("my_large")
    trainer.test(model=my_large_model, dataloaders=test_dataloader)
