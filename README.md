# Predictive Attractor Models
Official repository for the paper "[Predictive Attractor Models](https://arxiv.org/abs/2410.02430)" published at NeurIPS 2024.

<p align="center">
  <img src="assets/ssm.png" alt="Full architecture of this project"/>
</p>

---

## Environment Setup
1. Clone repository
```sh
git clone git@github.com:ramyamounir/pam.git
```


2. Create and activate conda environment 
```sh
conda create -n pamenv python=3.11 -y 
conda activate pamenv

# Install pytorch
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Use PamModel to learn and generate sequences
```python
from src.models.pam import PamModel
from src.utils.configs import get_pam_configs
from src.utils.exps import accuracy_SDR
from src.utils.sdr import SDR

# create two sequence
seq1 = [SDR(N=100, S=5) for _ in range(10)]
seq2 = [SDR(N=100, S=5) for _ in range(10)]

# Instantiate model with default parameters
pam = PamModel(N_c=100, N_k=4, W=5, **get_pam_configs())

# Learn two sequence
pam.learn_sequence(seq1)
pam.learn_sequence(seq2)

# generate the first sequence
recall = pam.recall_sequence_offline(seq1[0], len(seq1) - 1)

# Calculate accuracy (IoU)
print(accuracy_SDR(seq1, recall))
```

<p align="center">
  <img src="assets/gen.png" alt="Learning and generating a sequence"/>
</p>

--- 

## Paper Results Reproducibility





