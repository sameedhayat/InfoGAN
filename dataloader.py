from torch.utils.data import DataLoader
import torch.utils.data as utils
from torchvision import datasets, transforms
import numpy as np
import torch

# def get_data():
#     X = np.load('data/eeg/train_signal.npy')
#     #X = np.swapaxes(X, 1, 2)
#     X = X[0, :, :]
#     y = np.load('data/eeg/train_labels.npy')
#     # one hot encoding of the labels
#     y = np.eye(4)[y.reshape(-1)]
#
#     return X, y

def get_data():
    import os
    os.sys.path.append('/home/schirrmr/braindecode/code/braindecode/')
    from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
    from braindecode.datasets.bbci import BBCIDataset
    from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
    from braindecode.datautil.signalproc import exponential_running_standardize
    subject_id = 4  # 1-14
    loader = BBCIDataset('/data/schirrmr/schirrmr/HGD-public/reduced/train/{:d}.mat'.format(subject_id),
                         load_sensor_names=['C3'])
    cnt = loader.load()
    cnt = cnt.drop_channels(['STI 014'])
    from collections import OrderedDict
    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    # Here you can choose a larger sampling rate later
    # Right now chosen very small to allow fast initial experiments
    cnt = resample_cnt(cnt, new_fs=500)
    cnt = mne_apply(lambda a: exponential_running_standardize(a.T, factor_new=1e-3, init_block_size=1000, eps=1e-4).T,
                    cnt)
    ival = [0, 2000]  # ms to cut trial
    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset.X, dataset.y

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
        data_loader = utils.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader

    return data_loader