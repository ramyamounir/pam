import torch, torchvision
import torch.nn.functional as F
import os, json
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid

import sys; sys.path.append('./')
from src.models.pam import PamModel
from src.utils.sdr import SDR
from src.utils.exps import checkdir, accuracy_SDR, accuracy_POLAR, accuracy_BIN, set_seed
from src.utils.configs import get_pam_configs, get_pc_configs, get_hn_configs
from src.models.single_tpc import SingleLayertPC
from src.models.double_tpc import MultilayertPC
from src.models.mcahn import ModernAsymmetricHopfieldNetwork
from src.datasets.binary import generate_multiple_video_SDR_from_data, add_noise_SDR_patterns
from src.datasets.sparse_autoencoder import SparseAutoEncoder


def get_model(model, N, W, specs):

    if model.startswith('PAM'):
        net = PamModel(N_c=N, N_k=specs['N_k'], W=W, **specs['configs'])

    elif model.startswith('PC'):

        if specs['L'] == 1:
            net = SingleLayertPC(input_size=N, **specs['configs'])
        elif specs['L'] == 2:
            net = MultilayertPC(hidden_size=N*2, output_size=N, **specs['configs'])

    elif model.startswith('HN'):
        net = ModernAsymmetricHopfieldNetwork(input_size=N, **specs['configs'])

    return net


def train_model(seqs, net):

    if isinstance(net, PamModel):
        for seq in seqs:
            net.learn_sequence(seq)
    elif isinstance(net, SingleLayertPC) or isinstance(net, MultilayertPC):
        for seq in seqs:
            X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
            losses = net.train_seq(X_polar)
    elif isinstance(net, ModernAsymmetricHopfieldNetwork):
        pass
    else:
        raise ValueError("model not found")

def recall_seq(imgs, recons, seq, net, sae, query):

    if isinstance(net, PamModel):
        if query=='offline': recall = net.generate_sequence_offline(seq[0], len(seq)-1)
        elif query=='online': recall = net.generate_sequence_online(seq)
        iou = accuracy_SDR(seq[1:], recall[1:])
        recall_polar = torch.stack([x.bin.float() for x in recall])*2.0-1.0
        pred = sae.decode(recall_polar)
        pred = ((pred+1.0)/2.0)
        mse_imgs = F.mse_loss(imgs.cpu(), pred)
        mse_recons = F.mse_loss(recons.cpu(), pred)
    elif isinstance(net, SingleLayertPC) or isinstance(net, MultilayertPC):
        X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
        recall = net.recall_seq(X_polar, query)
        iou = accuracy_POLAR(X_polar[1:], recall[1:])
        pred = sae.decode(recall)
        pred = ((pred+1.0)/2.0)
        mse_imgs = F.mse_loss(imgs.cpu(), pred)
        mse_recons = F.mse_loss(recons.cpu(), pred)
    elif isinstance(net, ModernAsymmetricHopfieldNetwork):
        X_polar = torch.stack([x.bin.float() for x in seq])*2.0-1.0
        recall = net.recall_seq(X_polar, query)
        iou = accuracy_POLAR(X_polar[1:], recall[1:])
        pred = sae.decode(recall)
        pred = ((pred+1.0)/2.0)
        mse_imgs = F.mse_loss(imgs.cpu(), pred)
        mse_recons = F.mse_loss(recons.cpu(), pred)
    else:
        raise ValueError("model not found")

    return iou, mse_imgs, mse_recons, pred

def get_images(imgs):
    return make_grid((imgs.reshape(-1, *imgs.shape[-3:])*255.0).cpu().to(torch.uint8), nrow=10)


def compute(N, model, specs, num_seq, P, seed):

    # get model with seed
    set_seed(seed)
    W = int(specs['W']*N) if specs['W_type']=='percentage' else specs['W']
    net = get_model(model=model, N=N, W=W, specs=specs)

    # get data with seed
    set_seed(seed)
    data = torch.load(os.path.join(specs['data_dir'], 'data.pth'))
    imgs, sdrs, recons = generate_multiple_video_SDR_from_data(data, num_seq, P)

    sae = SparseAutoEncoder(N_c=N, W=W, img_height=imgs[0].shape[-2], img_width=imgs[0].shape[-1], n_channels=imgs[0].shape[-3])
    sae.load_state_dict(torch.load(os.path.join(specs['data_dir'], 'weights.pth')))

    # train
    train_model(sdrs, net)

    # eval
    results = {"input_imgs": get_images(imgs), "recon_imgs":get_images(recons)}
    imgs = imgs[0]
    sdrs = sdrs[0]
    recons = recons[0]

    iou, mse_imgs, mse_recons, pred = recall_seq(imgs, recons, sdrs, net, sae, 'online')

    results['iou'] = iou.item()
    results['mse_imgs'] = mse_imgs.item()
    results['mse_recons'] = mse_recons.item()
    results['pred_imgs'] = make_grid((pred*255.0).cpu().to(torch.uint8), nrow=P)

    return results




def main(save_base_dir, data_base_dir_5, data_base_dir_50, N, seed):

    models = ['PAM-8', 'HN-1-50', 'HN-2-50', 'PC-1', 'PC-2']

    specs = {
            'PAM-1':{'N_k':1, 'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_pam_configs()},
            'PAM-4':{'N_k':4, 'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_pam_configs()},
            'PAM-8':{'N_k':8, 'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_pam_configs()},
            'PAM-16':{'N_k':16, 'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_pam_configs()},
            'PAM-24':{'N_k':24, 'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_pam_configs()},
            'PC-1':{'L':1, 'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_pc_configs(l=1)},
            'PC-2':{'L':2, 'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_pc_configs(l=2)},
            'HN-1-5': {'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_hn_configs(d=1)},
            'HN-2-5': {'W':0.05, 'W_type':'percentage', 'data_dir':data_base_dir_5, 'configs': get_hn_configs(d=2)},
            'HN-1-50': {'W':0.5, 'W_type':'percentage', 'data_dir':data_base_dir_50, 'configs': get_hn_configs(d=1)},
            'HN-2-50': {'W':0.5, 'W_type':'percentage', 'data_dir':data_base_dir_50, 'configs': get_hn_configs(d=2)},
             }



    for model in models:
        print(f'Current model: {model}')


        results = compute(
                        N=N,
                        model=model, 
                        specs=specs[model],
                        num_seq=2,
                        P=10,
                        seed=seed,
                    )



        specific_save_dir = os.path.join(save_base_dir, model)
        os.makedirs(specific_save_dir, exist_ok=True)

        json.dump({'iou': results['iou'], 'mse_imgs':results['mse_imgs'], 'mse_recons':results['mse_recons']}, open(os.path.join(specific_save_dir, f'{str(seed).zfill(3)}.json'), 'w'))
        torchvision.io.write_png(results['input_imgs'], os.path.join(specific_save_dir, f'input_{str(seed).zfill(3)}.png'))
        torchvision.io.write_png(results['recon_imgs'], os.path.join(specific_save_dir, f'recons_{str(seed).zfill(3)}.png'))
        torchvision.io.write_png(results['pred_imgs'], os.path.join(specific_save_dir, f'preds_{str(seed).zfill(3)}.png'))




if __name__ == "__main__":


    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_001'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    for i in range(10):
        main(save_base_dir=os.path.join(save_base_dir, 'moving_mnist'), 
             data_base_dir_5=f'results/train_moving_mnist_sae/run_001', 
             data_base_dir_50=f'results/train_moving_mnist_sae/run_002', 
             N=100,
             seed=i)


