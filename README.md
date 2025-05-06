# CM-UNet

<p align="center">  
  <img src="https://github.com/user-attachments/assets/330c184e-4e79-4a8f-b6c5-71c7ebd9cf54" width="800" />  
</p>  


<p align="center">  
  <img src="https://github.com/user-attachments/assets/596189a3-28b4-4e62-8127-e746a94b5f82" width="400" />  
</p>  

<p align="center">  
  <img src="https://github.com/user-attachments/assets/b4359e71-e724-4b64-96eb-11d406022c05" width="400" />  
</p>  

<p align="center">  
  <img src="https://github.com/user-attachments/assets/4b03fa7c-df49-4239-ac61-fe20f56b45d3" width="400" />  
</p>  





# How to use

# Requirements
- NVIDIA GPU: 
  - Python 3.10.13
  - conda create -n your_env_name python=3.10.13  
- torch 2.1.1 + cu118  
  - pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118  

- Ubuntu 20.04  
- Pytorch 2.0.0  
- Python 3.8  
- CUDA 11.8
- an NVIDIA GeForce RTX 3090 GPU with 24 GB of memory.


# Training
For the training, you must run the `train.py` with your desired arguments. You need to change variables and arguments respectively. Below, you can find a brief description of the arguments.
```
--savedir  
The save address of the trained model file  

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

