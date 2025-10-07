import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import onnxruntime as ort
import numpy as np
import datas
import DeeperSiameseVae
from load_validation_data import get_validation_data

DEVICE = "cpu"
ONNX_DIR = "models/onnx"
CHECKPOINT_DIR = "models/sttorm_cd_lightning_ckpt"

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
MODEL_SIZES = ["small", "medium", "large"]
MODEL_TYPES = ["variable", "fixed"]
LATENT_DIM = 128
INPUT_SHAPE = (10, 32, 32)
TOLERANCE = 1e-4  # acceptable difference between ONNX and PyTorch outputs
# ---------------------------------------------------------------------


def load_model(checkpoint_path, size, initialized_metrics):
    """Load PyTorch model from checkpoint."""
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
        latent_dim=LATENT_DIM,
        input_shape=INPUT_SHAPE,
        initialized_metrics=initialized_metrics,
        learning_rate=0.001,
        weight_decay=0.001,
        margin_size=0.5,
        variable_margin=("variable" in checkpoint_path),
        log_all_metrics=False,
        dataset_valid=None,
        index=None,
        cos_baseline=False,
        export=True,
    )
    model.eval()
    model.to(DEVICE)
    return model


def run_onnx_inference(onnx_path, input_tensor):
    """Run inference using ONNX Runtime."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": input_tensor.numpy()}
    ort_outs = session.run(None, ort_inputs)
    return torch.tensor(ort_outs[0])


def compare_outputs(torch_output, onnx_output):
    """Compare PyTorch and ONNX model outputs numerically."""
    abs_diff = torch.abs(torch_output - onnx_output)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    return max_diff, mean_diff


if __name__ == "__main__":
    print("Loading validation data for metric initialization...")
    before_images, after_images, change_masks = get_validation_data(ravaen=False, disaster_type="floods")
    dataset = datas.ValidTestDataset(before_images, after_images, change_masks)
    initialized_metrics = dataset.get_initialized_metrics()

    print("\nStarting ONNX verification...\n")

    for model_type in MODEL_TYPES:
        for size in MODEL_SIZES:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{size}_{model_type}.ckpt")
            onnx_path = os.path.join(ONNX_DIR, f"change_detection_{size}_{model_type}_{INPUT_SHAPE[0]}_{INPUT_SHAPE[1]}_{INPUT_SHAPE[2]}.onnx")

            if not os.path.exists(onnx_path):
                print(f" Missing ONNX model: {onnx_path}")
                continue

            print(f"🔍 Testing {size} ({model_type}) ...")

            # Load PyTorch and ONNX models
            model = load_model(ckpt_path, size, initialized_metrics)
            dummy_input = torch.randn(1, *INPUT_SHAPE, device=DEVICE)

            with torch.no_grad():
                torch_output = model(dummy_input)
            onnx_output = run_onnx_inference(onnx_path, dummy_input.cpu())

            # Compare
            max_diff, mean_diff = compare_outputs(torch_output, onnx_output)
            passed = max_diff < TOLERANCE

            print(f"    Max diff:  {max_diff:.6f}")
            print(f"    Mean diff: {mean_diff:.6f}")
            print(f"    Match status: {'PASS' if passed else 'FAIL'}\n")

    print("Verification complete.")