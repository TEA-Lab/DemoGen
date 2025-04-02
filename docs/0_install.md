# Installing the Full Environment


## 0. Create conda Env
```bash
conda remove -n demogen --all
conda create -n demogen python=3.8
conda activate demogen
```

## 1. Install pip Packages 
Optionally using Tsinghua mirror by adding: `-i https://pypi.tuna.tsinghua.edu.cn/simple`
```bash
pip3 install torch==2.0.1 torchvision torchaudio 

pip3 install wandb ipdb gpustat visdom notebook mediapy torch_geometric natsort scikit-video easydict pandas moviepy imageio imageio-ffmpeg termcolor av open3d dm_control==1.0.8 dill==0.3.5.1 hydra-core==1.2.0 einops==0.4.1 diffusers==0.11.1 zarr==2.12.0 numba==0.56.4 pygame==2.1.2 shapely==1.8.4 tensorboard==2.10.1 tensorboardx==2.5.1 absl-py==0.13.0 pyparsing==2.4.7 jupyterlab==3.0.14 scikit-image yapf==0.31.0 opencv-python==4.5.3.56 psutil av matplotlib setuptools==59.5.0 kaleido==0.2.1 accelerate==0.33 timm==1.0.9 imagecodecs==2023.3.16 fpsample==0.3.3 transformers==4.45.1 huggingface-hub==0.24.6 pynput h5py
```

## 2. Install 𝑫𝒆𝒎𝒐𝑮𝒆𝒏
```bash
cd demo_generation
pip install -e .
cd ..
```

## 3. Install visuomotor policies (DP3 & DP_UMI)
```bash
cd diffusion_policies
pip install -e .
cd ..
```

## 4. Install point cloud visualizer
```bash
cd pcd_visualizer
pip install -e .
cd ..
```

