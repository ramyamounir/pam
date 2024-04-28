# solo-sdr
HTM prediction with associative memory

<p align="center">
  <img src="assets/full.png" alt="Full architecture of this project"/>
</p>

---

## Environment Setup
1. Create a conda environment with `conda create -n pamenv python=3.11 -y` and activate it with `conda activate pamenv`

--- 

## Usage


```python
from src.utils.sdr import SDR
from src.models.pam import PamModel
from src.utils.configs import get_pam_configs

seq = [SDR(N=100, S=5) for _ in range(10)]
print(seq)
pam = PamModel(N_c=100, N_k=4, W=5, **get_pam_configs())

pam.learn_sequence(seq)
recall = pam.recall_sequence_offline(seq[0], len(seq)-1)
print(recall)
```

