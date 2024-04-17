import torch
import numpy as np
import random
import os, shutil, subprocess

def accuracy_SDR(gt, recall):
    assert len(gt) == len(recall), "Groundtruth and Recall are not the same size"

    avg_iou = 0.0
    for g, r in zip(gt, recall):
        g_sparsity, r_sparsity = len(g)/g.N, len(r)/r.N
        expected_iou = (g_sparsity*r_sparsity)/(g_sparsity+r_sparsity-(g_sparsity*r_sparsity))
        avg_iou += ((g.iou(r)-expected_iou)/(1.0-expected_iou))

    return avg_iou/len(gt)


def accuracy_POLAR(gt, recall):
    assert len(gt) == len(recall), "Groundtruth and Recall are not the same size"

    avg_iou = 0.0
    for g, r in zip((gt+1.0)/2.0, (recall+1.0)/2.0):

        g_sparsity, r_sparsity = sum(g)/len(g), sum(r)/len(r)
        expected_iou = (g_sparsity*r_sparsity)/(g_sparsity+r_sparsity-(g_sparsity*r_sparsity))
        iou = torch.sum(torch.logical_and(g, r))/torch.sum(torch.logical_or(g, r))
        avg_iou += (iou - expected_iou)/(1.0 - expected_iou)

    return avg_iou/len(gt)

def accuracy_BIN(gt, recall):
    assert len(gt) == len(recall), "Groundtruth and Recall are not the same size"

    avg_iou = 0.0
    for g, r in zip(gt, recall):

        g_sparsity, r_sparsity = sum(g)/len(g), sum(r)/len(r)
        expected_iou = (g_sparsity*r_sparsity)/(g_sparsity+r_sparsity-(g_sparsity*r_sparsity))
        iou = torch.sum(torch.logical_and(g, r))/torch.sum(torch.logical_or(g, r))
        avg_iou += (iou - expected_iou)/(1.0 - expected_iou)


    return avg_iou/len(gt)


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class ImagesDatset(torch.utils.data.Dataset):
    def __init__(self, images): self.images = images
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], torch.tensor([0])


def video_extract(video_input, video_output, snippet_size, height, width, output_ext):

    video_output = os.path.join(video_output, f'frame_%10d.{output_ext}')
    os.makedirs(os.path.dirname(video_output), exist_ok=True)

    command = f'\
        ffmpeg -i {video_input}\
        -vf "fps={int(1.0/snippet_size)},\
        scale={width}:{height}"\
        -vsync vfr {video_output}\
        -loglevel panic\
        > /dev/null 2>&1'

    process = subprocess.Popen(command, shell=True)
    process.communicate()


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
