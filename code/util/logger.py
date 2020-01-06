#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging as absl_logging

import logging

import coloredlogs

################################################################################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '../data', help='the data directory')
flags.DEFINE_string('result_dir', '../results', help='the result directory')

flags.DEFINE_string('data_npy_dir', '../data/npy',
                    help='the numpy data directory')
flags.DEFINE_string('result_model_dir', '../results/model',
                    help='the model directory')
flags.DEFINE_string('result_figure_dir', '../results/figure',
                    help='the figure directory')

################################################################################################################################

# fmt='%(asctime)s %(filename)12.12s:%(lineno)-4d %(levelname)8s %(message)s'
fmt = '%(asctime)s %(levelname)8s %(message)s'
datefmt = '%Y-%m-%d %H:%M:%S'
field_styles = {
    'asctime': {'color': 'green'},
    'filename': {'color': 'cyan'},
    'levelname': {'color': 'magenta', 'bold': True},
}
level_styles = {
    'debug': {'color': 'blue'},
    'info': {'color': 'green'},
    'warning': {'color': 'yellow'},
    'error': {'color': 'red', 'bold': True},
    'critical': {'background': 'red', 'bold': True},
}

formatter = coloredlogs.ColoredFormatter(
    fmt=fmt,
    datefmt=datefmt,
    level_styles=level_styles,
    field_styles=field_styles,
)
absl_logging.set_verbosity(logging.DEBUG)
absl_logging.get_absl_handler().setFormatter(formatter)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.handlers.clear()
mpl_logger.setLevel(logging.WARNING)
