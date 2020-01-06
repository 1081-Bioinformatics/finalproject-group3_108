#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import app
from absl import flags
from absl import logging

import os
import csv

import numpy as np

import util.logger
from util.data_loader import DataLoader

################################################################################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string('input_name', '0_original', help='the name of input data')

flags.DEFINE_string('output_name', None, help='the name of output data')
flags.mark_flag_as_required('output_name')

flags.DEFINE_spaceseplist('features', None, help='the selected features')
flags.mark_flag_as_required('features')

################################################################################################################################

def main(_):

    if FLAGS.log_dir:
        os.makedirs(FLAGS.log_dir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(FLAGS.task, FLAGS.log_dir)

    logging.debug('Arguments:')
    for k, v in FLAGS.flag_values_dict().items():
        logging.debug(f'- {k}: {v}')

    idir = os.path.join(FLAGS.data_npy_dir, FLAGS.input_name)
    loader = DataLoader(idir)

    idxs = []
    for f in FLAGS.features:
        i = np.argwhere(loader.feature == f)
        assert len(i) >= 1, f'Unknown feature "{f}"!'
        assert len(i) <= 1, f'Duplicated feature "{f}"!'
        idxs.append(i[0, 0])
    idxs = np.asarray(idxs)

    loader.feature = loader.feature[idxs]
    loader.train_set.x = loader.train_set.x[:, idxs]
    loader.test_set.x = loader.test_set.x[:, idxs]

    odir = os.path.join(FLAGS.data_npy_dir, FLAGS.output_name)
    loader.dump(loader.feature, loader.train_set, loader.test_set, output_dir=odir)


if __name__ == '__main__':
    app.run(main)
