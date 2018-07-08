from torch.utils.data import DataLoader
import torch.utils.data as utils
from torchvision import datasets, transforms
import numpy as np
import torch

def get_data():
    X = np.load('data/eeg/train_signal.npy')
    #X = np.swapaxes(X, 1, 2)

    y = np.load('data/eeg/train_labels.npy')
    # one hot encoding of the labels
    y = np.eye(4)[y.reshape(-1)]

    return X, y

def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'eeg':
        X, y = get_data()

        tensor_x = torch.stack([torch.Tensor(i) for i in X])  # transform to torch tensors
        tensor_y = torch.stack([torch.Tensor(i) for i in y])

        my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
        data_loader = utils.DataLoader(my_dataset)  # create your dataloader

    return data_loader