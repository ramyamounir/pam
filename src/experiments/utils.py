import torch
import os, shutil


def to_np(x):
    return x.cpu().detach().numpy()

def to_torch(x, device):
    return torch.from_numpy(x).to(device)


def checkdir(path, careful=False):
    if os.path.exists(path):
        if careful: return False
        shutil.rmtree(path)
    os.makedirs(path)
    return True
