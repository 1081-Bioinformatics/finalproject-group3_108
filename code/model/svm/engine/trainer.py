#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import os
import pickle

import numpy as np

from sklearn.svm import LinearSVC as Model

from engine.data_loader import DataLoader
from engine.scorer import Scorer

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################


class Trainer:

    def __init__(self):

        self.train_data, self.test_data = DataLoader.load(
            input_dir=FLAGS.input_dir,
        )

        self.scorer = Scorer()

        self.model = Model(
            max_iter=FLAGS.num_iter,
        )

    ################################################################

    def run_train(self):

        x, y = self.train_data
        self.model.fit(x, y)

        self._save_model()

    def run_test(self):
        self._load_model()

        x, y_true = self.test_data
        y_score = self.model.predict(x)
        self.scorer(
            y_true=y_true,
            y_score=y_score,
        )

    ########################################################################################################################
    # Routines model I/O

    def _load_model(self):
        """Load pretrained model.
        """
        file = FLAGS.model_file
        logging.info(f'Loading model from {file} ...')
        with open(file, 'rb') as fin:
            self.model = pickle.load(fin)

    def _save_model(self):

        file = FLAGS.model_file
        logging.info(f'Saving model into {file} ...')
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'wb') as fout:
            pickle.dump(self.model, fout)
