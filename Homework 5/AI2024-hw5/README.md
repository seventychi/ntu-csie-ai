# Homework 5

[//]: # (## Install Necessary Packages)

[//]: # ()
[//]: # (conda create -n hw5 python=3.11 -y)

[//]: # (conda activate hw5)

[//]: # (pip install -r requirements.txt)

## 建立 conda env

conda create -n ntu-ai-hw5-v2-env python=3.11 -y
conda activate ntu-ai-hw5-v2-env

## 安裝套件

pip install opencv-python==4.8.1.78
pip install swig==4.2.1
pip install gymnasium==0.29.1
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install numpy==1.26.4
pip install matplotlib==3.8.4
pip install imageio-ffmpeg
pip install imageio==2.34.1
pip install tqdm

## 安裝 pytorch gpu 版本 (已去信與 TA 確認可安裝)

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

## 安裝額外套件（已去信與 TA 確認可安裝）

pip install chardet charset_normalizer --upgrade

## Training command

```powershell
python pacman.py --save_root "./models"
```

## Evaluation command

```powershell
python pacman.py --eval --eval_model_path "./submissions/pacman_dqn.pt"
```