import torch
import os

import sys; sys.path.append('./')
from src.models.single_tpc import SingleLayertPC
from src.models.double_tpc import MultilayertPC
from src.models.mcahn import ModernAsymmetricHopfieldNetwork
from src.models.pam import PamModel
from src.models.pam_utils import SDR
from src.experiments.utils import checkdir
from src.data.cifar import load_sequence_cifar


def single_tpc(save_base_dir):
    tpc = SingleLayertPC(input_size=3072, lr=2e-5, data_type='continuous', learn_iters=400).cuda()

    seq = load_sequence_cifar(4, 32).cuda()
    seq = seq.reshape((32, 3072)) # seq_lenx3072
    # _plot_recalls(seq, "memory.png")


    losses = tpc.train_seq(seq)
    rec = tpc.recall_seq(seq, query='online')
    print(rec.shape, save_base_dir)
    # _plot_recalls(rec, "recall.png")

def multi_tpc():


    mtpc = MultilayertPC(
                 hidden_size=1024, 
                 output_size=3072, 
                 learn_lr=1e-5, 
                 inf_lr=1e-2, 
                 learn_iters=200,
                 inf_iters=100,
                 ).cuda()

    seq = load_sequence_cifar(4, 32).cuda()
    seq = seq.reshape((32, 3072)) # seq_lenx3072
    # _plot_recalls(seq, "memory.png")

    losses = mtpc.train_seq(seq)
    rec = mtpc.recall_seq(seq, query='online')
    # _plot_recalls(rec, "recall.png")


def modern_ahn():

    hop = ModernAsymmetricHopfieldNetwork(3072, data_type='continuous', sep='softmax').cuda()

    seq = load_sequence_cifar(4, 32).cuda()
    seq = seq.reshape((32, 3072)) # seq_lenx3072
    # _plot_recalls(seq, "memory.png")

    rec = hop.recall_seq(seq, query='online')
    # _plot_recalls(rec, "mahn.png")

def pam():
    pam_model = PamModel(128, 4, 10, 0.8, 0.0, 1.0, 100)

    seq = [SDR(128, 10) for _ in range(10)]

    print('gt:')
    for s in seq:
        print(s)

    # train
    pam_model.train_seq(seq)

    print('\nonline predicted:')
    rec = pam_model.recall_seq(seq, 'online')
    for r in rec:
        print(r)


    print('\noffline predicted:')
    rec = pam_model.recall_seq(seq, 'offline')
    for r in rec:
        print(r)



if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_001'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    # single_tpc(save_base_dir)
    # pam()

