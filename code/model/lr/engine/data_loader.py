#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import os
from collections import namedtuple

import numpy as np

from sklearn import model_selection
from sklearn import preprocessing

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################

DataSet = namedtuple('DataSet', ('x', 'y',))


class DataLoader:

    @classmethod
    def load(cls, input_dir):

        train_x, train_y = cls._load_data(os.path.join(input_dir, 'train'))
        test_x, test_y = cls._load_data(os.path.join(input_dir, 'test'))

        # scaler = preprocessing.StandardScaler()
        # train_x = scaler.fit_transform(train_x, train_y)
        # test_x = scaler.transform(test_x)

        return DataSet(train_x, train_y), DataSet(test_x, test_y)

    @classmethod
    def _load_data(cls, input_prefix):

        xfile = f'{input_prefix}_x.npy'
        yfile = f'{input_prefix}_y.npy'

        x = np.load(xfile)
        y = np.load(yfile)

        return x, y
