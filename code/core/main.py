#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

import os

from absl import app
from absl import flags
from absl import logging

import util.logger

################################################################################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string('input_name', None, help='the input data name')
flags.mark_flag_as_required('input_name')

flags.DEFINE_string('model_name', None, help='the model name')
flags.mark_flag_as_required('model_name')

flags.DEFINE_string('full_name', None, help='<model_name>_<input_name>')

flags.DEFINE_string('input_dir', None, help='<data_npy_dir>/<input_name>')
flags.DEFINE_string('model_file', None, help='<result_model_dir>/<full_name>')
flags.DEFINE_string('figure_file', None, help='<result_figure_dir>/<full_name>')

flags.DEFINE_string('device', None, help='the device to run')

flags.DEFINE_bool('no_train', False, help='Don\'t training.')
flags.DEFINE_bool('no_test', False, help='Don\'t testing.')

################################################################################################################################

def get_trainer():
    model_name = FLAGS.model_name

    if model_name in ['attention']:
        from core.network.trainer import Trainer
        return Trainer()
    else:
        from core.statistics.trainer import Trainer
        return Trainer()

    logging.fatal(f'Unknown model {model_name}!')

################################################################################################################################


def main(_):

    if FLAGS.log_dir:
        os.makedirs(FLAGS.log_dir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(FLAGS.task, FLAGS.log_dir)

    logging.debug('Arguments:')
    for k, v in FLAGS.flag_values_dict().items():
        logging.debug(f'- {k}: {v}')

    # Set paths
    FLAGS.full_name   = f'{FLAGS.model_name}_{FLAGS.input_name}'
    FLAGS.input_dir   = os.path.join(FLAGS.data_npy_dir, FLAGS.input_name)
    FLAGS.model_file  = os.path.join(FLAGS.result_model_dir, FLAGS.full_name)
    FLAGS.figure_file = os.path.join(FLAGS.result_figure_dir, FLAGS.full_name)

    # Run
    trainer = get_trainer()
    if not FLAGS.no_train:
        logging.info("==== Training ====")
        trainer.run_train()

    if not FLAGS.no_test:
        logging.info("==== Testing ====")
        trainer.run_test()

if __name__ == '__main__':
    app.run(main)
