## Installation

```powershell
# 建立 conda 環境
conda create -y -n ai_hw6 python=3.10

# activate env
conda activate ai_hw6

# install pytorch on cuda 12.1 and relevant packages, press Y if processing needs
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 cudatoolkit xformers -c pytorch -c nvidia -c xformers

# install unsloth and relevant packages, press Y if processing need
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# install tqdm packaging wandb
pip install tqdm packaging wandb

# validate installation (can skip)
nvcc --version
python -m xformers.info
python -m bitsandbytes 
```

## Finetune

Model: unsloth/mistral-7b-v0.3-bnb-4bit

Hyper parameter:

在原始的 DPO 及 ORPO 都採用預設的 hyper-parameters，並未額外控制超參數。在 extra experiments 中則針對 DPO 定義以下的 hyper parameters (原因可參考 hw6_d09922009.pdf)

| hyper parameter  | default | fine-tuned |
| ---------------- | ------- | ---------- |
| train_batch_size | 2       | 4          |
| eval_batch_size  | 2       | 4          |
| max_steps        | 0       | 1500       |
| num_epochs       | 1       | 3          |
| weight_decay     | 0       | 0.01       |
| warmup_ratio     | 0       | 0.1        |

執行以下 command 進行訓練 (建議調整為自己的 wandb token:

```powershell
# for default DPO experiment
bash run.sh DPO unsloth/mistral-7b-v0.3-bnb-4bit 61471e9edd3765f3c45812e0d5afff00373eef88

# for default ORPO experiment
bash run.sh ORPO unsloth/mistral-7b-v0.3-bnb-4bit 61471e9edd3765f3c45812e0d5afff00373eef88

# for extra experiments
python main.py --exp_name "DPO" --model_name "unsloth/mistral-7b-v0.3-bnb-4bit" --train --wandb_token "61471e9edd3765f3c45812e0d5afff00373eef88" --train_batch_size 4 --eval_batch_size 4 --max_steps 1500 --weight_decay 0.01 --warmup_ratio 0.1 --num_epochs 3
```
