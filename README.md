# CTI-UNet: Hybrid Local Features and Global Representations Efficiently


## Abstract

Recent advancements in medical image segmentation have demonstrated superior performance by combining Transformer and U-Net due to the Transformer's exceptional ability to capture long-range semantic dependencies. However, existing approaches mostly replace or concatenate the Convolutional Neural Networks (CNNs) and Transformers in series, which limits the potential of their combination. In this paper, we introduce a dual-branch feature encoder, CTI-UNet, that effectively fuses the global representations and local features of the CNN and Transformer branches at different scales through bidirectional feature interaction. Our proposed method outperforms existing approaches on multiple medical datasets, demonstrating state-of-the-art performance.



## Data Preparation
We train and test our models on three datasets, Synapse, ACDC, and GlaS, where Synapse is a pre-processed dataset from [TransUNet](), which we do not have the right to disclose, please contact them to obtain the dataset.

Create a new folder `data`，then the directory structure is as follows:

```shell
data/synapse
├── annotations
│   ├── train
│   └── val
└── images
    ├── train
    └── val
```

## Usage

We use the [MMSegmentation]() framework to build our model. Please refer to the instructions of MMSegmentation to install the runtime environment.

To train the model, run:

```sh
# Synapse
bash tools/dist_train.sh local_configs/cti_unet/cti-unet-aux_l4_i4_2x12_224x224_80k_synapse_casa_pos.py 2

# ACDC
bash tools/dist_train.sh local_configs/cti_unet/cti-unet-aux_l4_i4_2x12_224x224_80k_acdc_casa_pos.py 2

#Glas
python tools/train.py local_configs/cti_unet/cti-unet-aux_l4_i4_1x2_224x224_80k_glas_casa_pos.py
```

All training logs will be saved in `./work_dirs` folder by default.
