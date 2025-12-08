import torch

checkpoint_path = "/original_checkpoints/B_train_VAE_128large/3nmy7jtc/checkpoints/epoch_00-step_29653.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# dict_keys(['state_dict', 'callbacks', 'optimizer_states', 'lr_schedulers', 'native_amp_scaling_state', 'hparams_name', 'hyper_parameters'])
state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
checkpoint["state_dict"] = state_dict
new_path = "models/ravaen_modified_ckpt/large_modified_checkpoint.ckpt"
torch.save(checkpoint, new_path)
