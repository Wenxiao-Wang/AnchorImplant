# AnchorImplant
AnchorImplant: Precise Mandibular Implant Pose Prediction via Anatomical-Anchored Registration and nnf-UNet

## How to Use
- Download and configure [**nnUNet**](https://github.com/MIC-DKFZ/nnUNet)

- Move **nnUNetTrainer_fUNet.py** to **.../nnUNet/nnunetv2/training/nnUNetTrainer/** of the configured nnUNet

- Use nnf-UNet just like nnUNet:

> Data Preprocessing
```
nnUNetv2_plan_and_preprocess -d DATASET_NAME_OR_ID --verify_dataset_integrity
```

> Training
```
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres 0 -tr nnUNetTrainer_fUNet
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres 1 -tr nnUNetTrainer_fUNet
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres 2 -tr nnUNetTrainer_fUNet
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres 3 -tr nnUNetTrainer_fUNet
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres 4 -tr nnUNetTrainer_fUNet
```

> Best Configuration
```
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c 3d_lowres -f 0 1 2 3 4 -tr nnUNetTrainer_fUNet
```

> Testing
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c 3d_lowres -tr nnUNetTrainer_fUNet
```
# Dataset
To the tasks of IAN registration, implant transformation and generation, we collected 300 preoperative and postoperative CBCT scans(https://cbm.dhu.edu.cn/_upload/tpl/0e/ff/3839/template3839/page7_1_1.html). Specifically, all CBCT volumes were resampled to a uniform spacing of 0.5×0.5×0.5 mm
