import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms


def load_sequence_cifar(seed, seq_len):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the CIFAR10 dataset
    cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of digits
    sequence = torch.zeros((seq_len, 3, 32, 32))

    # Sample `seq_len` random images from the MNIST dataset
    indices = torch.randint(0, len(cifar), (seq_len,))
    for i, idx in enumerate(indices):
        img, _ = cifar[idx]
        sequence[i] = img

    return sequence
