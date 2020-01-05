#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'
__title__ = 'LDA'

import os

from logging import Formatter

from absl import app
from absl import flags
from absl import logging

from engine.trainer import Trainer

################################################################################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string('appname', __title__, help='app name')

flags.DEFINE_integer('num_iter', 100000, help='the maximum number of itertion')

flags.DEFINE_string('input_dir', None,
                    help='the path to the input data files.')
flags.mark_flag_as_required('input_dir')

flags.DEFINE_string('model_file', None,
                    help='the path to the model file')
flags.DEFINE_string('figure_file', None,
                    help='the path to the figure file')

flags.DEFINE_bool('no_train', False,
                  help='Do not training.')

flags.DEFINE_bool('no_test', False,
                  help='Do not testing.')

################################################################################################################################


def main(_):

    # Init logger
    formatter = Formatter(
        # fmt='%(asctime)s %(filename)12.12s:%(lineno)-4d %(levelname)8s | %(message)s',
        fmt='%(asctime)s %(levelname)8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.get_absl_handler().setFormatter(formatter)

    if FLAGS.log_dir:
        os.makedirs(FLAGS.log_dir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(FLAGS.task, FLAGS.log_dir)

    logging.debug('Arguments:')
    for k, v in FLAGS.flag_values_dict().items():
        logging.debug(f'- {k}: {v}')

    # Init trainer
    trainer = Trainer()

    if not FLAGS.no_train:
        logging.info("==== Training ====")
        trainer.run_train()

    if not FLAGS.no_test:
        logging.info("==== Testing ====")
        trainer.run_test()


if __name__ == '__main__':
    app.run(main)
