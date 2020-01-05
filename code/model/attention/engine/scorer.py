#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

from sklearn.metrics import accuracy_score

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################


class Scorer:

    def __call__(self, *, y_true, y_pred):

        # AUC
        print(accuracy_score(y_true, y_pred))


################################################################################################################################
