# AnchorImplant
AnchorImplant: Precise Mandibular Implant Pose Prediction via Anatomical-Anchored Registration and nnf-UNet

## How to Use
- Download and configure [**nnUNet**](https://github.com/MIC-DKFZ/nnUNet)

- Move **nnUNetTrainer_fUNet.py** to **.../nnUNet/nnunetv2/training/nnUNetTrainer/** of the configured nnUNet

- Use nnf-UNet just like nnUNet:

> Data Preprocessing
```
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity
```

> Training
```
nnUNetv2_train 502 3d_lowres 0 -tr nnUNetTrainer_fUNet
nnUNetv2_train 502 3d_lowres 1 -tr nnUNetTrainer_fUNet
nnUNetv2_train 502 3d_lowres 2 -tr nnUNetTrainer_fUNet
nnUNetv2_train 502 3d_lowres 3 -tr nnUNetTrainer_fUNet
nnUNetv2_train 502 3d_lowres 4 -tr nnUNetTrainer_fUNet
```

> Best Configuration
```
nnUNetv2_find_best_configuration 502 -c 3d_lowres -f 0 1 2 3 4 -tr nnUNetTrainer_fUNet
```

> Testing
```
nnUNetv2_predict -i .../nnUNetFrame/nnUNet_raw/Dataset502_your_dataset/imagesTs/ -o .../your_predict_path/ -d 502 -c 3d_lowres -tr nnUNetTrainer_fUNet
```
