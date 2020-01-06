#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import os
import tqdm

import numpy as np
import torch

from util.data_loader import DataLoader

from sklearn import model_selection

################################################################################################################################

FLAGS = flags.FLAGS

BATCH_SIZE = 32

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
        )

        dataset = torch.utils.data.TensorDataset(self.x, self.y)
        self.loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
        )

    @property
    def num_sample(self):
        return self.x.shape[0]

    @property
    def num_feature(self):
        return self.x.shape[1]

    def __len__(self):
        return int(np.ceil(self.num_sample / BATCH_SIZE))

    def __iter__(self):
        yield from self.loader

    @classmethod
    def load(cls, *, input_dir, device):

        loader = DataLoader(input_dir, scale=True)

        train_x = loader.train_set.x
        train_y = loader.train_set.y

        test_x = loader.test_set.x
        test_y = loader.test_set.y

        train_x, dev_x, train_y, dev_y = model_selection.train_test_split(
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
