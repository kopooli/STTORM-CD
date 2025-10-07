import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import datas
import DeeperSiameseVae
from load_validation_data import get_validation_data

DEVICE = "cpu"
ONNX_DIR = "models/onnx"
os.makedirs(ONNX_DIR, exist_ok=True)

# Option to export with dynamic axes (batch dimension variable)
USE_DYNAMIC_AXES = True


def load_model(checkpoint_path, size, initialized_metrics, dataset=None, index=None, cos_baseline=False):
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
        index=index,
        cos_baseline=cos_baseline,
        export=True,
    )
    model.to(DEVICE)
    model.eval()
    return model


def export_to_onnx(model, model_type, size, use_dynamic_axes=USE_DYNAMIC_AXES):
    in_c, in_h, in_w = model.input_shape
    model_name = f"change_detection_{size}_{model_type}_{in_c}_{in_h}_{in_w}.onnx"
    onnx_path = os.path.join(ONNX_DIR, model_name)
    dummy_input = torch.randn(1, *model.input_shape, device=DEVICE)
    print(model(dummy_input))

    if use_dynamic_axes:
        dynamic_axes = {"input": {0: "batch_size"}, "latent": {0: "batch_size"}}
    else:
        dynamic_axes = None

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported {model_name} to {onnx_path}")


if __name__ == "__main__":
    my_checkpoint_path = "models/sttorm_cd_lightning_ckpt/"
    original_path = "models/ravaen_modified_ckpt/"

    # Dummy dataset just for initialized_metrics
    before_images, after_images, change_masks = get_validation_data(ravaen=False, disaster_type="floods")
    dataset = datas.ValidTestDataset(before_images, after_images, change_masks)
    initialized_metrics = dataset.get_initialized_metrics()

    # All models to export
    models = {
        # STTORM-CD variable
        "variable": {
            "small": load_model(os.path.join(my_checkpoint_path, "small_variable.ckpt"), "small", initialized_metrics),
            "medium": load_model(os.path.join(my_checkpoint_path, "medium_variable.ckpt"), "medium", initialized_metrics),
            "large": load_model(os.path.join(my_checkpoint_path, "large_variable.ckpt"), "large", initialized_metrics),
        },
        # STTORM-CD fixed
        "fixed": {
            "small": load_model(os.path.join(my_checkpoint_path, "small_fixed.ckpt"), "small", initialized_metrics),
            "medium": load_model(os.path.join(my_checkpoint_path, "medium_fixed.ckpt"), "medium", initialized_metrics),
            "large": load_model(os.path.join(my_checkpoint_path, "large_fixed.ckpt"), "large", initialized_metrics),
        },
    }

    # Export all models with new naming convention
    for model_type, sizes in models.items():
        for size, model in sizes.items():
            export_to_onnx(model, model_type, size, use_dynamic_axes=USE_DYNAMIC_AXES)

    print("\nAll models exported to", ONNX_DIR)