#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import os
import pickle

import numpy as np

from util.data_loader import DataLoader
from util.scorer import Scorer

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################

def get_model():
    model_name = FLAGS.model_name

    if model_name == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()

    if model_name == 'lr':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()

    if model_name == 'svm':
        from sklearn.svm import LinearSVC
        return LinearSVC(max_iter=100000)

    if model_name == 'nb':
        from sklearn.naive_bayes import BernoulliNB
        return BernoulliNB()

    if model_name == 'ft':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier()

    if model_name == 'rndfor':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()

    if model_name == 'elasticnet':
        from sklearn.linear_model import ElasticNet
        return ElasticNet()

    logging.fatal(f'Unknown model {model_name}!')

################################################################################################################################

class Trainer:

    def __init__(self):

        self.loader = DataLoader(FLAGS.input_dir)
        self.scorer = Scorer()
        self.model  = get_model()

        self.model_file = f'{FLAGS.model_file}.pkl'

    ################################################################

    def run_train(self):

        x = self.loader.train_set.x
        y = self.loader.train_set.y

        logging.info(f'train: X = {x.shape}')
        logging.info(f'train: Y = {y.shape}')

        self.model.fit(x, y)

        self._save_model()

    def run_test(self):
        self._load_model()

        x = self.loader.test_set.x
        y = self.loader.test_set.y

        logging.info(f'test: X = {x.shape}')
        logging.info(f'test: Y = {y.shape}')

        y_score = self.model.predict(x)
        self.scorer(
            y_true=y,
            y_score=y_score,
        )

    ########################################################################################################################
    # Routines model I/O

    def _load_model(self):
        """Load pretrained model.
        """
        ifile = self.model_file
        logging.info(f'<< {ifile} ...')
        with open(ifile, 'rb') as fin:
            self.model = pickle.load(fin)

    def _save_model(self):

        ofile = self.model_file
        logging.info(f'>> {ofile} ...')
        os.makedirs(os.path.dirname(ofile), exist_ok=True)
        with open(ofile, 'wb') as fout:
            pickle.dump(self.model, fout)
