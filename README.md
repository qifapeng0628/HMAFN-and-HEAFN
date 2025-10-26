# HMAFN-and-HEAFN

This repository contains the official PyTorch implementation for GMD-AFNet (Version 2.0), an improved version of the paper presented at the International Conference on Intelligent Computing (ICIC 2025). The code for Version 1.0 is no longer provided.

## 1. Environment Setup

This codebase was developed and tested in the following environment:
* **OS**: Ubuntu 22.04
* **Python**: 3.9
* **CUDA**: 11.6

### Installation

1.  **Create and activate a virtual environment** (recommended):
    ```bash
    python3.9 -m venv venv
    source venv/bin/activate
    ```

2.  **Install PyTorch (cu116)**:
    Install the PyTorch version matching CUDA 11.6.
    ```bash
    # This command installs PyTorch 1.13.1, torchvision 0.14.1, and torchaudio 0.13.1
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url [https://download.pytorch.org/whl/cu116](https://download.pytorch.org/whl/cu116)
    ```

3.  **Install other dependencies**:
    You can save the following content as `requirements.txt` and run `pip install -r requirements.txt`.
    
    ```text
    # requirements.txt
    numpy==1.26.4
    opencv-python==4.9.0.80
    tifffile==2024.8.30
    timm==1.0.19
    tqdm==4.66.2
    ```
    

## 2. Dataset Preparation

* **MIMIC-CXR-JPG**: Can be downloaded from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
* **CXR-Eye Dataset**: To download the CXR-Eye dataset, please check the [gaze_repo](https://github.com/cxr-eye-gaze/eye-gaze-dataset).

After downloading, please unzip the datasets to your specified directory and modify the data paths in the code accordingly.

## 3. Pretrained Weights

* **ViT (Vision Transformer)**: The ViT pretrained weights (`vit-base-patch16-224-in21k`) used in this project can be downloaded from [Hugging Face (Click here)](https://https://huggingface.co/google/vit-base-patch16-224-in21k/tree/main).

## 4. Usage

### Training

Taking HMAFN as an example:
```bash
cd /code/GMDAFNetv2/HMAFN/CXR-Eye
python main.py
```
## 5. Model Checkpoints

* **Best Model**: [Click here to download our trained best model](https://pan.baidu.com/s/1Xfj3KKn4Hzm7xOkMIR-xMQ?pwd=fpgw)

## 6. Acknowledgements

We thank the following repos which provided helpful functions and datasets:
* [CUB-GHA](https://github.com/yaorong0921/CUB-GHA/tree/main)
* [eye-gaze-dataset](https://github.com/cxr-eye-gaze/eye-gaze-dataset)
