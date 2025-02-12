# MotionAura-ICLR-2025
Official source code of "MotionAura: Generating High-Quality and Motion Consistent Videos using Discrete Diffusion", published in ICLR 2025 (Spotlight)

## Paper
You can read the full paper [here](https://arxiv.org/abs/2410.07659).

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8 or higher (for GPU support)
- [PyTorch](https://pytorch.org/get-started/locally/) 2.0.0 or higher


### Clone the Repository
```bash

git clone https://github.com/yourusername/MotionAura-ICLR-2025.git

cd MotionAura-ICLR-2025

conda create -n motionaura python=3.8

conda activate motionaura

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

pip install -e .

