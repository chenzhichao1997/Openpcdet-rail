# RailVoxelDet

**RailVoxelDet** is a lightweight 3D object detection framework tailored for railway transportation scenarios. It is built upon the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) open-source codebase with extensive modifications to support railway-specific datasets and tasks.

---

## 🔗 Reference

Our work is developed based on the official [OpenPCDet v0.6.0](https://github.com/open-mmlab/OpenPCDet) repository:

> OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds  
> [GitHub Repository](https://github.com/open-mmlab/OpenPCDet)  
> Shaoshuai Shi*, Chaoxu Guo*, Li Jiang, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li (* equal contribution)

Please cite their work if you use this project.

---

## 📁 Dataset

You can download the customized railway dataset from Baidu Netdisk:

- File: `railway_dataset`
- Link: [https://pan.baidu.com/s/19Cen11luxb_AaL8FCxsDLA?pwd=uj9p](https://pan.baidu.com/s/19Cen11luxb_AaL8FCxsDLA?pwd=uj9p)  
- Extraction Code: `uj9p`  

After downloading, unzip the dataset and place it under the `data/railway_dataset/` directory.

---

## 🔧 Installation

Please follow the official OpenPCDet [installation instructions](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md). A summarized version is provided below:

### 1. Environment Preparation

```bash
conda create -n railvoxeldet python=3.8 -y
conda activate railvoxeldet

### 2. Install PyTorch and CUDA
# Example for CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

### 3. Compile C++/CUDA ops
python setup.py develop

Ensure the structure looks like:
RailVoxelDet/
├── pcdet/               # Customized detection modules
├── tools/               # Training and inference scripts
├── data/
│   └── railway_dataset/ # Place dataset here
├── requirements.txt
├── setup.py
└── README.md

