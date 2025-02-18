### STTORM-CD
![Germany Floods - Inference](germany_floods_inferece.png)
This repository contains code for STTORM-CD research, tested with Python 3.11 and 3.12. To run it successfully, install the dependencies from requirements.txt using pip and execute all scripts from within this repository (i.e., after cloning this repository, run cd STTORM-CD before running any Python scripts). The code is designed with predefined paths for quick setup.
#### Dataset
The **annotated dataset** used for this research can be accessed on **[Zenodo](https://doi.org/10.5281/zenodo.14891438)**. Additionally, the **[RaVAEn](https://github.com/spaceml-org/RaVAEn) dataset** is also used and can be downloaded **[here](https://drive.google.com/drive/folders/1VEf49IDYFXGKcfvMsfh33VSiyx5MpHEn?usp=sharing)**.

After downloading, extract the **Zenodo dataset** into the `data/` directory, ensuring the following structure: `./data/dataset/train/italia_emilia_romagna/1/mask.npy`

For the **RaVAEn dataset**, unzip it into the `data/ravaen/` directory, ensuring the structure is like: `./data/ravaen/floods`

#### Models
The fine-tuned STTORM-CD models are stored as checkpoints in the `models/sttorm_cd_lightning_ckpt` directory. RaVAEn models, converted into a compatible format for use with the provided code, are saved in  
`models/ravaen_modified_ckpt`.  

Additionally, the `onnx` and `tvm` folders contain deployed models, following the naming convention: `change_detection_ModelSize_InputName_InputShape_TargetDevice`. For example: `change_detection_large_images_1_10_32_32_q8_a53.so`.  

The `tvm` folder also includes `.py` scripts for measuring runtime. These scripts are designed to be executed on the target device but require the `tvm` library to be installed.

#### Training
The model was trained using the script `sweep_script.py`, this script utilize Weights and Biases, and you need to set ENV variable `WANDB_API_KEY`. Alternatively, you can train it without WandB by using the simple `train_script.py`.

#### Testing
To test the models, use the `testing.py` script. In this file, you can specify the disaster type or choose the flood testing dataset. You can toggle between testing with metrics or generating heatmaps, as well as switch between the cosine baseline, indices, or model by modifying the variables at the top of the `DeeperSiameseVae.py` file.

#### Visualization
If you're interested in the metrics, check out the `RaVAEn_tile_wise_AUPRC_limitations_showcase.ipynb`. This notebook presents a fictional example from the STTORM-CD Supplementary Information that illustrates the limitations in RaVAEn's use of metrics.
