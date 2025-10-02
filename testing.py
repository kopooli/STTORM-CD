import os
import json
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datas
import DeeperSiameseVae
from load_validation_data import get_validation_data

DEVICE = "cpu"
BASE_LOG_DIR = "test_metrics_logs"
os.makedirs(BASE_LOG_DIR, exist_ok=True)


def load_model(checkpoint_path, size, initialized_metrics, dataset):
    latent_dim = 128
    input_shape = (10, 32, 32)
    if size == "large":
        hidden_channels = [32, 64, 128]
        extra_depth = 2
    elif size == "medium":
        hidden_channels = [32, 64, 128]
        extra_depth = 0
    elif size == "small":
        hidden_channels = [16, 32, 64]
        extra_depth = 0
    else:
        raise ValueError(f"Unknown model size: {size}")

    model = DeeperSiameseVae.DeeperVAE.load_from_checkpoint(
        checkpoint_path,
        map_location=DEVICE,
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


def run_tests(models, dataloader, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    trainer = pl.Trainer(accelerator=DEVICE, max_epochs=1)

    for model_name, model in models.items():
        print(f"Testing {model_name} on {log_dir} ...")
        results = trainer.test(model=model, dataloaders=dataloader)

        # Save metrics to JSON
        json_path = os.path.join(log_dir, f"{model_name}_test_metrics.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved test metrics to {json_path}")


if __name__ == "__main__":
    my_checkpoint_path = "models/sttorm_cd_lightning_ckpt/"
    original_path = "models/ravaen_modified_ckpt/"

    disasters = [
        ("landslides", True, "landslides"),
        ("hurricanes", True, "hurricanes"),
        ("floods_ravaen", True, "floods"),
        ("floods_sttorm", False, "floods"),
        ("fires", True, "fires"),
    ]

    for folder_name, ravaen_flag, disaster_type in disasters:
        print(f"\n=== Running tests for {folder_name} (ravaen={ravaen_flag}) ===")

        # Prepare dataset
        before_images, after_images, change_masks = get_validation_data(
            ravaen=ravaen_flag, disaster_type=disaster_type
        )
        test_dataset = datas.ValidTestDataset(before_images, after_images, change_masks)
        initialized_metrics = test_dataset.get_initialized_metrics()

        # Dataloader
        n_cpu = os.cpu_count()
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(n_cpu / 2),
            persistent_workers=True,
        )

        # Load models
        models = {
            "original_small": load_model(os.path.join(original_path, "small_modified_checkpoint.ckpt"), "small", initialized_metrics, test_dataset),
            "original_medium": load_model(os.path.join(original_path, "medium_modified_checkpoint.ckpt"), "medium", initialized_metrics, test_dataset),
            "original_large": load_model(os.path.join(original_path, "large_modified_checkpoint.ckpt"), "large", initialized_metrics, test_dataset),
            "my_small_variable": load_model(os.path.join(my_checkpoint_path, "small_variable.ckpt"), "small", initialized_metrics, test_dataset),
            "my_medium_variable": load_model(os.path.join(my_checkpoint_path, "medium_variable.ckpt"), "medium", initialized_metrics, test_dataset),
            "my_large_variable": load_model(os.path.join(my_checkpoint_path, "large_variable.ckpt"), "large", initialized_metrics, test_dataset),
            "my_small_fixed": load_model(os.path.join(my_checkpoint_path, "small_fixed.ckpt"), "small", initialized_metrics, test_dataset),
            "my_medium_fixed": load_model(os.path.join(my_checkpoint_path, "medium_fixed.ckpt"), "medium", initialized_metrics, test_dataset),
            "my_large_fixed": load_model(os.path.join(my_checkpoint_path, "large_fixed.ckpt"), "large", initialized_metrics, test_dataset),
        }

        # Run tests
        log_dir = os.path.join(BASE_LOG_DIR, folder_name)
        run_tests(models, test_dataloader, log_dir)

    print("\nAll disasters tested and metrics saved to", BASE_LOG_DIR)