from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import os.path as osp
import svhn_config as config
config = config.config

import torch
from torch.utils.data import Dataset

import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, data,id =None,label=None, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.id = id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cur_img = self.data[index]
            
        if self.transform is not None:
            img = self.transform(cur_img)
        if self.label is not None :
            cur_label = self.label[index]
            return img, cur_label
        elif id is None:
            return img, cur_label
        else:
            return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order
            # comment it to be order-agnostic
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """Evenly sample seq_len items from num items."""
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid



class RescaledCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            size=(224,224)
    ) -> None:

        super(RescaledCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set


        if self.train:
            downloaded_list = self.train_list
            string = "train"
        else:
            downloaded_list = self.test_list
            string = "test"

        self.data: Any = []
        self.targets = []


        if not os.path.exists(os.path.join(self.root, self.base_folder,f"rescaled_{size}_{string}.npy")):
            print("Rescaled dataset does not exist, creating it at ", os.path.abspath(os.path.join(self.root, self.base_folder,f"rescaled_{size}_{string}.npy")))
            self.prepare_rescaled(downloaded_list=downloaded_list, size=size)
        else:
            print("Rescaled dataset exists, loading it ", os.path.abspath(os.path.join(self.root, self.base_folder,f"rescaled_{size}_{string}.npy")))
            self.data = np.load(os.path.join(self.root, self.base_folder,f"rescaled_{size}_{string}.npy"))
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])

        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def prepare_rescaled(self, size=None, downloaded_list=None):
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = transforms.Resize(size)(torch.tensor(self.data)).numpy()
        if self.train:
            string = "train"
        else:
            string = "test"
        print("Saving the dataset at ")
        np.save(os.path.join(self.root, self.base_folder,f"rescaled_{size}_{string}.npy"),self.data)

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        # if not check_integrity(path, self.meta['md5']):
        #     raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                        ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)


    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



class RescaledCIFAR100(RescaledCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }