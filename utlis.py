import contextlib
from hashlib import new
import os
import copy
from re import X
import torch
import pickle
import codecs
import pandas
import logging
import numpy as np
from collections import namedtuple
import csv
import pathlib
import PIL

from functools import partial
from PIL import Image
from torchvision import datasets, transforms
from typing import Any, Callable, List, Optional, Union, Tuple, Dict, cast

import warnings
from PIL import Image
import os.path
from urllib.error import URLError
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity, download_url
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import make_dataset
from torchvision.transforms import functional as FF

def fill_param(param):
    if param["dataset"] == "cifar10":
        param["num_classes"] = 10


def get_data(param):
    if param["dataset"] == "cifar10":
        dst_test = CIFAR10_BADNETS('../ddbd/dataset-distillation/data', train=False, download=True,
                    transform=None, trigger_label=0, portion=0, backdoor_size=0, backdoor=False, clean_test=True)
        x_test = []
        y_test = []
        for img, label in dst_test:
            x_test.append(np.array(img))
            y_test.append(label)
        # x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        # y_train = y_train.astype(np.long)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
        y_test = y_test.astype(np.long)
        return x_test, y_test.reshape((-1,)), x_test, y_test.reshape((-1,))
    if param["dataset"] == "stl10":
        dst_test = STL10_BADNETS('../ddbd/dataset-distillation/data', split="test", download=True,
                    transform=None, trigger_label=0, portion=0, backdoor_size=0, backdoor=False, clean_test=True)
        x_test = []
        y_test = []
        for img, label in dst_test:
            x_test.append(np.array(img))
            y_test.append(label)
        # x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        # y_train = y_train.astype(np.long)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
        y_test = y_test.astype(np.long)
        return x_test, y_test.reshape((-1,)), x_test, y_test.reshape((-1,))

    if param["dataset"] == "svhn":
        dst_test = SVHN_BADNETS('../ddbd/dataset-distillation/data', split="test", download=True,
                    transform=None, trigger_label=0, portion=0, backdoor_size=0, backdoor=False, clean_test=True)
        x_test = []
        y_test = []
        for img, label in dst_test:
            x_test.append(np.array(img))
            y_test.append(label)
        # x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        # y_train = y_train.astype(np.long)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        #x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
        y_test = y_test.astype(np.long)
        return x_test, y_test.reshape((-1,)), x_test, y_test.reshape((-1,))

    if param["dataset"] == "fmnist":
        dst_test = FashionMNIST_BADNETS('../ddbd/dataset-distillation/data', train=False, download=True,
                    transform=None, trigger_label=0, portion=0, backdoor_size=0, backdoor=False, clean_test=True)
        x_test = []
        y_test = []
        for img, label in dst_test:
            x_test.append(np.array(img))
            y_test.append(label)
        # x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        # y_train = y_train.astype(np.long)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        #x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
        y_test = y_test.astype(np.long)
        return x_test, y_test.reshape((-1,)), x_test, y_test.reshape((-1,))


def poison(x_train, y_train, param):
    if param["poisoning_method"] == "badnet":
        x_train, y_train = _poison_badnet(x_train, y_train, param)
    return x_train, y_train


def _poison_badnet(x_train, y_train, param):
    target_label = param["target_label"]
    for i in range(x_train.shape[0]):
        for c in range(3):
            for w in range(3):
                for h in range(3):
                    x_train[i][c][-(w+2)][-(h+2)] = 255
        y_train[i] = target_label
    return x_train, y_train


class STL10_BADNETS(VisionDataset):

    base_folder = "stl10_binary"
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
    class_names_file = "class_names.txt"
    folds_list_file = "fold_indices.txt"
    train_list = [
        ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
        ["train_y.bin", "5a34089d4802c674881badbb80307741"],
        ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
    ]

    test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"], ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        trigger_label: int = 0,
        portion: float =0.1,
        backdoor_size: int = 2,
        backdoor: bool = True,
        clean_test: bool = True,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)

        self.trigger_label = trigger_label
        self.portion = portion
        self.backdoor = backdoor
        self.clean_test = clean_test
        self.backdoor_size = backdoor_size

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        if self.split == "train":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)

        elif self.split == "train+unlabeled":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate((self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == "unlabeled":
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

    def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = "Value for argument folds should be in the range [0, 10), but got {}."
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = FF.resize(img, (32,32))
        img = np.asarray(img)

        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]

    def __loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = os.path.join(self.root, self.base_folder, labels_file)
            with open(path_to_labels, "rb") as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, "rb") as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds: Optional[int]) -> None:
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds) as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=" ")
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]

class SVHN_BADNETS(VisionDataset):
    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        trigger_label: int = 0,
        portion: float =0.1,
        backdoor_size: int = 2,
        backdoor: bool = True,
        clean_test: bool = True,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        self.trigger_label = trigger_label
        self.portion = portion
        self.backdoor_size = backdoor_size
        self.backdoor = backdoor
        self.clean_test = clean_test

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class CIFAR10_BADNETS(VisionDataset):
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
            trigger_label: int = 0,
            portion: float =0.1,
            backdoor_size: int = 2,
            backdoor: bool = True,
            clean_test: bool = True,
    ) -> None:

        super(CIFAR10_BADNETS, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        self.trigger_label = trigger_label
        self.portion = portion
        self.backdoor = backdoor
        self.clean_test = clean_test
        self.backdoor_size = backdoor_size

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

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
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        
        self.targets = np.array(self.targets)

        if self.backdoor:
            if not self.train:
                if self.clean_test:
                    self.portion = 0
                else:
                    self.portion = 1

            self._add_trigger()

        ''''
        self.bad_data, self.bad_targets = self._add_trigger()

        self.total_data = np.concatenate((self.data,self.bad_data),0)
        self.total_targets = np.concatenate((self.targets,self.bad_targets),0)
        '''

    def _add_trigger(self):
        '''
        Based on Vera Xinyue Shen Badnets https://github.com/verazuo/badnets-pytorch
        '''
        perm = np.random.permutation(len(self.data))[0: int(len(self.data) * self.portion)]
        width, height, _ = self.data.shape[1:]
        # self.data[perm, width-3, height-3, :] = 255
        # self.data[perm, width-3, height-2, :] = 255
        # self.data[perm, width-2, height-3, :] = 255
        # self.data[perm, width-2, height-2, :] = 255

        # assert self.backdoor_size == 4

        self.data[perm, width-self.backdoor_size-1:width-1, height-self.backdoor_size-1:height-1, :] = 255
        self.targets[perm] = self.trigger_label
        
        '''

        new_data = self.data[perm]
        new_targets = self.targets[perm]

        new_data[:, width-3, height-3, :] = 255
        new_data[:, width-3, height-2, :] = 255
        new_data[:, width-2, height-3, :] = 255
        new_data[:, width-2, height-2, :] = 255

        new_targets[:] = self.trigger_label

        '''

        # logging.info("Injecting Over: %d Bad Imgs" % len(perm))
        # return new_data, new_targets

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
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

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

class MNIST_BADNETS(VisionDataset):
    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            trigger_label: int = 0,
            portion: float =0.01,
            backdoor_size: int = 2,
            backdoor: bool = True,
            clean_test: bool = True,     
    ) -> None:
        super(MNIST_BADNETS, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        self.trigger_label = trigger_label
        self.portion = portion
        self.backdoor_size = backdoor_size
        self.backdoor = backdoor
        self.clean_test = clean_test

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets = self._load_data()
        if self.backdoor:
            if not self.train:
                if self.clean_test:
                    self.portion = 0
                else:
                    self.portion = 1

            self.perm = np.random.permutation(len(self.data))[0: int(len(self.data) * self.portion)]

    # def _add_trigger(self):
    #     perm = np.random.permutation(len(self.data))[0: int(len(self.data) * self.portion)]

    #     width, height = self.data.shape[1:]

    #     self.data[perm, width-self.backdoor_size-1:width-1, height-self.backdoor_size-1:height-1] = 255
    #     self.targets[perm] = self.trigger_label


    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = datasets.mnist.read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = datasets.mnist.read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        img = FF.resize(img, (32,32))
        if self.backdoor:
            if index in self.perm:
                img = np.asarray(img)
                width, height = img.shape
                img[width-self.backdoor_size-1:width-1, height-self.backdoor_size-1:height-1] = 255
                img = Image.fromarray(img)
                target = self.trigger_label

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    logging.info("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    logging.info(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class FashionMNIST_BADNETS(MNIST_BADNETS):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]