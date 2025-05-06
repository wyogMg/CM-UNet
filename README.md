# CM-UNet
CM-UNet adopts an asymmetric U-shaped encoder-decoder structure. It can be described as an encoder layer, a decoder layer, a bottleneck layer, and skip-connection. As our core component, HCME is used as the foundational module for the encoder layer, decoder layer, and bottleneck layer.
<p align="center">  
  <img src="https://github.com/user-attachments/assets/221e1021-027a-4d9a-9c2f-c59f51ab7390" width="800" />  
</p>  


This module introduces an efficient long-range dependency modeling approach that combines the principles of Vision Mamba with channel attention mechanisms. It captures global context effectively while maintaining linear complexity, enabling better feature relevance suppression.

<p align="center">  
  <img src="https://github.com/user-attachments/assets/cfdaae17-cb7b-469a-b707-0211118a79fb" width="400" />  
</p>  

Designed to preserve fine-grained crack details, this module employs parameter-efficient large separable convolutional kernels to enhance local feature interaction and reduce the texture degradation often seen in state-space models.

<p align="center">  
  <img src="https://github.com/user-attachments/assets/b288f617-da17-48fa-9999-396ff19fd6e3" width="300" />  
</p>  


This module dynamically fuses features from multiple scales using a learnable adaptive weighting mechanism. Integrated into the skip connections, it bridges the semantic gap between encoder and decoder, improving the overall feature representation capability of the network.

<p align="center">  
  <img src="https://github.com/user-attachments/assets/8c254392-4ed4-4c1c-844a-d75666297a98" width="400" />  
</p>  



# How to use

# Requirements
- Ubuntu 20.04  
- Pytorch 2.0.0  
- Python 3.8  
- torch 2.0.0 + cu118  
  - pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
- Hardware Spec
  - an NVIDIA GeForce RTX 3090 GPU with 24 GB of memory


```
torch @ http://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp38-cp38-linux_x86_64.whl
torchvision @ http://download.pytorch.org/whl/cu118/torchvision-0.15.1%2Bcu118-cp38-cp38-linux_x86_64.whl
mamba-ssm==1.1.4
pandas==2.0.3
causal-conv1d==1.1.3.post1
torch==2.0.0+cu118  
torchvision==0.15.1+cu118  
timm==0.9.12  
einops==0.7.0  
numpy==1.24.2  
matplotlib==3.7.1  
Pillow==10.4.0  
opencv-python==4.10.0.84
tqdm==4.66.4
einops==0.7.0
PyWavelets==1.4.1
```

# Training
For the training, you must run the `train.py` with your desired arguments. You need to change variables and arguments respectively. Below, you can find a brief description of the arguments.
```
--savedir  
Path to save the trained model file  

--imgdir  
Path to the dataset images  

--labdir  
Path to the dataset labels  

--imgsz  
Input image size for the network  

--filename  
Path to save the loss values  
```
# Inference
For inference, you need to run the `test.py`. Most of the parameters are like for the `train.py`.

# Citation
```
@article{
  title={The U-shaped crack segmentation network based on channel-Mamba enhancement and multi-scale aggregation},  
  journal={The Visual Computer},  
  author={Yongming Wang, Shigang Hu, Guoyi Zheng, Jianxin Wang} 
}
```

