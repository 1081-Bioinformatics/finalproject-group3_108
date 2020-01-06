#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import app
from absl import flags
from absl import logging

import os
import csv

import numpy as np
from sklearn import model_selection

import util.logger
from util.data_loader import DataLoader, DataSet

################################################################################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'mirbase/mirna_tpm.csv', help='the CSV file name')
flags.DEFINE_string('output_name', '0_original', help='the output directory')
flags.DEFINE_integer('seed', 0, help='the random seed')

################################################################################################################################

def main(_):

    if FLAGS.log_dir:
        os.makedirs(FLAGS.log_dir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(FLAGS.task, FLAGS.log_dir)

    logging.debug('Arguments:')
    for k, v in FLAGS.flag_values_dict().items():
        logging.debug(f'- {k}: {v}')

    ifile = os.path.join(FLAGS.data_dir, FLAGS.input_file)
    logging.info(f'<< {ifile} ...')
    with open(ifile) as fin:
        reader = csv.reader(fin)
        data = np.asarray([row for row in reader])
        assert data[0, 0] == 'name'
        assert data[0, -1] == 'label'

    feature = data[0,  1:-1].astype(np.str_)
    name    = data[1:, 0   ].astype(np.str_)
    x       = data[1:, 1:-1].astype(np.float32)
    y       = data[1:,   -1].astype(np.float32)

    x_train, x_test, y_train, y_test, name_train, name_test = \
        model_selection.train_test_split(
            x, y, name,
            test_size=0.25,
            random_state=FLAGS.seed,
        )

    train_set = DataSet(x=x_train, y=y_train, name=name_train)
    test_set = DataSet(x=x_test, y=y_test, name=name_test)

    odir = os.path.join(FLAGS.data_npy_dir, FLAGS.output_name)
    DataLoader.dump(feature, train_set, test_set, output_dir=odir)

if __name__ == '__main__':
    app.run(main)
