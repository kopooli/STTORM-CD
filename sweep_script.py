import wandb
import os
from save_tiles_with_npy_mask import tile_original_images
from preprocess_train_tiles import get_anchor_positive_negative_tiles
from load_validation_data import get_validation_data
from train_script import train_model


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        final_tiles_path = f"data/tiled/final_{config.margin_size[0]}_{config.margin_size[1]}_{config.stride}"
        if not os.path.exists(final_tiles_path):
            tile_original_images(config.margin_size, config.stride)
            get_anchor_positive_negative_tiles(final_tiles_path)
        train_model(
            model_size=config.model_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            margin_size=config.margin_size[0],
            variable_margin=config.variable_margin,
            batch_size=config.batch_size,
            log_all_metrics=False,
            n_epochs=config.epochs,
            before_images=before_images,
            after_images=after_images,
            change_masks=change_masks,
            final_tiles_path=final_tiles_path,
        )


if __name__ == "__main__":
    # Check if the API key is set in the environment variables
    api_key = os.getenv("WANDB_API_KEY")

    if api_key is None:
        # If API key is not set, prompt the user to set it manually
        print("WANDB_API_KEY environment variable is not set. Please set it manually.")
        exit()

    # Log in using the API key
    wandb.login(key=api_key)
    sweep_config = {
        "method": "grid",
        "metric": {"name": "area_under_the_curve_avg_overall", "goal": "maximize"},
    }
    parameters_dict = {
        "batch_size": {"value": 32},  # [16, 32, 64, 128]
        "weight_decay": {"value": 0.000005},  # [0.0, 0.0001, 0.00001, 0.000001]
        "learning_rate": {"value": 0.000001},  # [0.000003, 0.000002, 0.000001]
        "variable_margin": {"value": True},  # [True, False]
        "margin_size": {
            "value": (
                0.0,
                1,
            )  # [(0.0, 0.001), (0.25, 1), (0.5, 1), (0.75, 1), (0.0, 1)]
        },
        "stride": {"value": 32},  # [4, 8, 16, 32]
        "model_size": {"value": "medium"},  # , "medium", "large"
        "epochs": {"value": 1},
    }
    before_images, after_images, change_masks = get_validation_data(
        ravaen=True, disaster_type="floods"
    )
    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="mac_sweep")
    wandb.agent(sweep_id, train)
