#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import os
import tqdm

import numpy as np
import torch

from sklearn.model_selection import train_test_split

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################


class Batcher:

    def __init__(self, x, y, *, name, device, shuffle):

        logging.info(f'{name}: X = {x.shape}')
        logging.info(f'{name}: Y = {y.shape}')

        self.name = name
        self.x = torch.tensor(
            x,
            device=device,
            dtype=torch.float,
        )
        self.y = torch.tensor(
            y,
            device=device,
            dtype=torch.float,
        ).unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(self.x, self.y)
        self.loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=FLAGS.batch_size,
            shuffle=shuffle,
        )

    def __len__(self):
        return self.y.shape[0]

    def __iter__(self):
        yield from self.loader

    @classmethod
    def load(cls, *, input_dir, device):

        train_x, train_y = cls._load_data(os.path.join(input_dir, 'train'))
        test_x, test_y = cls._load_data(os.path.join(input_dir, 'test'))

        train_x, dev_x, train_y, dev_y = train_test_split(
            train_x,
            train_y,
            test_size=0.1,
            random_state=0,
        )

        train_batcher = Batcher(
            train_x,
            train_y,
            name='train',
            device=device,
            shuffle=True,
        )

        dev_batcher = Batcher(
            dev_x,
            dev_y,
            name='dev',
            device=device,
            shuffle=False,
        )

        test_batcher = Batcher(
            test_x,
            test_y,
            name='test',
            device=device,
            shuffle=False,
        )

        return train_batcher, dev_batcher, test_batcher

    @classmethod
    def _load_data(cls, input_prefix):

        xfile = f'{input_prefix}_x.npy'
        yfile = f'{input_prefix}_y.npy'

        x = np.load(xfile)
        y = np.load(yfile)

        return x, y
