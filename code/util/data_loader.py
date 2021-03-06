#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Load numpy format data and create training/testing datasets"""

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import os
from collections import namedtuple

import numpy as np

from sklearn import preprocessing

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################


class DataSet:

    def __init__(self, *, x, y, name):
        self.x = x
        self.y = y
        self.name = name


class DataLoader:

    def __init__(self, input_dir, scale=False):

        logging.info(f'<< {input_dir} ...')

        try:
            self.feature = np.load(os.path.join(input_dir, 'feature.npy'))
        except FileNotFoundError:
            self.feature = None

        self.train_set = self._load_data(os.path.join(input_dir, 'train'))
        self.test_set = self._load_data(os.path.join(input_dir, 'test'))

        if scale:
            self.scaler = preprocessing.StandardScaler()
            self.train_set.x = self.scaler.fit_transform(self.train_set.x)
            self.test_set.x = self.scaler.fit_transform(self.test_set.x)

    # Load dataset
    def _load_data(self, subdir):

        xfile = f'{subdir}/x.npy'
        yfile = f'{subdir}/y.npy'
        nfile = f'{subdir}/name.npy'

        try:
            x = np.load(xfile)
        except ValueError:
            x = np.load(xfile, allow_pickle=True).astype(np.float32)

        try:
            y = np.load(yfile)
        except ValueError:
            y = np.load(yfile, allow_pickle=True).astype(np.float32)

        try:
            name = np.load(nfile)
        except FileNotFoundError:
            name = None

        return DataSet(x=x, y=y, name=name)

    # Dump all data
    @classmethod
    def dump(cls, feature, train_set, test_set, *, output_dir):
        cls._dump_npy(feature, 'feature', output_dir)

        cls._dump_npy(train_set.name, 'train/name', output_dir)
        cls._dump_npy(train_set.x, 'train/x', output_dir)
        cls._dump_npy(train_set.y, 'train/y', output_dir)

        cls._dump_npy(test_set.name, 'test/name', output_dir)
        cls._dump_npy(test_set.x, 'test/x', output_dir)
        cls._dump_npy(test_set.y, 'test/y', output_dir)

    # Dump a numpy file
    def _dump_npy(data, name, output_dir):
        ofile = os.path.join(output_dir, f'{name}.npy')
        os.makedirs(os.path.dirname(ofile), exist_ok=True)

        logging.info(f'>> {ofile} ...')
        logging.debug(f'{data.dtype} {data.shape}')
        np.save(ofile, data, allow_pickle=False)
