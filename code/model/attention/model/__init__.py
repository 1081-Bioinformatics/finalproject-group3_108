#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import torch
import torch.nn.functional as F

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################


class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()
