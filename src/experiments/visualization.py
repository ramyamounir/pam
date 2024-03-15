from src.experiments.utils import to_np
import matplotlib.pyplot as plt


def _plot_recalls(recall, name):
    seq_len = recall.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(recall[j].reshape((3, 32, 32)).permute(1, 2, 0)))
        ax[j].axis('off')
        ax[j].set_aspect("auto")

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    # plt.savefig(fig_path + f'/{model_name}_len{seq_len}_query{args.query}', bbox_inches='tight', dpi=200)
    plt.savefig(f'./{name}', bbox_inches='tight', dpi=200)
