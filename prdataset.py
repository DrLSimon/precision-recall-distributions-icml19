from random import shuffle
from PIL import Image
import numpy as np
import torch

import os
import os.path
import tqdm
import hashlib

from torch.utils.data import Dataset, DataLoader


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(folder, extensions):
    images = []
    folder = os.path.expanduser(folder)
    for root, _, fnames in sorted(os.walk(folder)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_inception_embedding(directory, cache_dir='/tmp/prd_cache'):
    directory = os.path.realpath(directory)
    hash = hashlib.md5(directory.encode('utf-8')).hexdigest()
    path = os.path.join(cache_dir, hash + '.npy')
    if os.path.exists(path) and (os.stat(path).st_mtime > os.stat(directory).st_mtime):
        embeddings = np.load(path)
        return torch.from_numpy(embeddings)

class FolderDataset(Dataset):
    def __init__(self, folder, extensions, loader, transform):
        self.samples = make_dataset(folder, extensions)
        self.folder = folder
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.transform(self.loader(path))
        return sample

    def __len__(self):
        return len(self.samples)
        

class SourceTargetDataset(Dataset):
    def __init__(self, source_folder, target_folder, features=None, loader=pil_loader,
                 extensions=['jpg', 'png'],
                 transform_train=None, transform_test=None):
        self.loader = loader
        self.extensions = extensions
        self.source_samples = make_dataset(source_folder, extensions)
        self.target_samples = make_dataset(target_folder, extensions)
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.transform_train = transform_train
        self.transform_test = transform_test

        if features == 'tf':
            self.source_features = load_inception_embedding(source_folder)
            self.target_features = load_inception_embedding(target_folder)
            shuffle(self.source_features)
            shuffle(self.target_features)
        else:
            self.source_features= self.target_features = None
            shuffle(self.source_samples)
            shuffle(self.target_samples)
            if len(self.source_samples) != len(self.target_samples):
                raise ValueError(
                  'Length of source samples %d must be identical to length of '
                  'target samples %d.'
                  % (len(self.source_samples), len(self.target_samples)))
        nsamples = len(self)
        self.coin_flips =  torch.from_numpy(np.random.binomial(1, 0.5, size=[nsamples])).float()
        if nsamples == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.train()

    def precomputeFeatures(self, net, batch_size, device):
        self.source_features = self.__loadOrPrecomputeFeatures(self.source_folder, net, batch_size, device, self.transform_test)
        self.target_features = self.__loadOrPrecomputeFeatures(self.target_folder, net, batch_size, device, self.transform_test)

    def __precompute(self, folder, net, batch_size, device, transform):
        features = []
        dataset = FolderDataset(self.source_folder, self.extensions, self.loader, self.transform_test)
        dataset = FolderDataset(folder, self.extensions, self.loader, transform)
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
        for sample in tqdm.tqdm(loader, leave=False):
            feats = net(sample.to(device))
            features.append(feats)
        return torch.cat(features)

    def __loadOrPrecomputeFeatures(self, folder, net, batch_size, device, transform, cache_dir='/tmp/prd_cache_torch/'):
        hash = hashlib.md5(folder.encode('utf-8')).hexdigest()
        path = os.path.join(cache_dir, hash + '.npy')
        if os.path.exists(path) and (os.stat(path).st_mtime > os.stat(folder).st_mtime):
            embeddings = torch.from_numpy(np.load(path))
            return embeddings
        embeddings = self.__precompute(folder, net, batch_size, device, transform)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(path, 'wb') as f:
            embeddings = embeddings.cpu().numpy()
            np.save(f, embeddings)
        return torch.from_numpy(embeddings)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, flip) 
        """

        flip = self.coin_flips[index]

        if self.source_features is None:
            transform = self.transform_train
            if not self.is_train:
                flip = not flip
                transform = self.transform_test

            if flip:
                path = self.target_samples[index]
            else:
                path = self.source_samples[index]

            sample = self.loader(path)

            if transform is not None:
                sample = transform(sample)
        else:
            if not self.is_train:
                flip = not flip
            if flip:
                sample = self.target_features[index]
            else:
                sample = self.source_features[index]

        return sample, flip

    def __len__(self):
        return len(self.source_samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Source Location: {}\n'.format(self.source_folder)
        fmt_str += '    Target Location: {}\n'.format(self.target_folder)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

