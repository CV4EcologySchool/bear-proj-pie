# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import os.path as osp
import tarfile
import zipfile
from .data_manipulation import read_image, download_url


class ImageDataset(object):
    """An abstract class representing a Dataset.

    Args:
        train (list): contains tuples of (img_path(s), pid).
        test (list): contains tuples of (img_path(s), pid).
        transform: transform function.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        mode (str): 'train' or 'test'.
        verbose (bool): show information.
    """

    def __init__(
        self, train, test, transform=None, k_tfm=1, mode='train', verbose=True, **kwargs
    ):

        self.train = train
        self.test = test
        self.transform = transform
        self.k_tfm = k_tfm
        self.mode = mode
        self.verbose = verbose

        self.num_train_pids, self.num_train_seq_ids = self.get_num_pids(self.train)

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'test':
            self.data = self.test
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | test ]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        img_path, pid, seq_id = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {'img': img, 'pid': pid, 'impath': img_path}
        return item

    def __len__(self):
        return len(self.data)

    def get_num_pids(self, data):
        """Returns the number of training person identities.
        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        pids = set([items[1] for items in data])
        seq_ids = set([items[2] for items in data])
        return len(pids), len(seq_ids)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(self.__class__.__name__)
            )

        print('Creating directory "{}"'.format(dataset_dir))
        os.makedirs(dataset_dir, exist_ok=True)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(self.__class__.__name__, dataset_dir)
        )
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except Exception:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def _transform_image(self, tfm, k_tfm, img0):
        """Transforms a raw image (img0) k_tfm times with
        the transform function tfm.
        """
        img_list = []

        for k in range(k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

    def show_summary(self):
        num_train_pids, num_train_seq_ids = self.get_num_pids(self.train)
        num_test_pids, num_test_seq_ids = self.get_num_pids(self.test)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # seq_ids | # images ')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:5d}     | {:8d} '.format(num_train_pids, num_train_seq_ids, len(self.train)))
        print('  test     | {:5d} | {:5d}     | {:8d} '.format(num_test_pids, num_test_seq_ids, len(self.test)))
        print('  ----------------------------------------')
