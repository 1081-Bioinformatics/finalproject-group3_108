#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import json
import os
import tqdm

import numpy as np
import torch

from engine.batcher import Batcher
from engine.scorer import Scorer
from model import Model

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################


class Trainer:

    def __init__(self):

        if FLAGS.device is None:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(FLAGS.device)

        self.train_batcher, self.dev_batcher, self.test_batcher = \
            Batcher.load(
                input_dir=FLAGS.input_dir,
                device=self.device
            )

        self.scorer = Scorer()

        self.model = Model()

    ################################################################

    def run_train(self):

        epoch_idx_best = 0
        dev_loss_best = np.inf

        # Loop for epoch
        for epoch_idx in range(1, FLAGS.num_epoch + 1):

            # Run training and validation
            train_loss = self._run_train_epoch(self.train_batcher)
            dev_loss = self._run_eval_epoch(self.dev_batcher)

            # Check if reached current best
            if dev_loss < dev_loss_best:
                dev_loss_best = dev_loss
                epoch_idx_best = epoch_idx
                self._save_model()

            logging.info(
                f'Epoch {epoch_idx}/{FLAGS.num_epoch} | '
                f'Train Loss: {train_loss:9.6f} | '
                f'Dev Loss: {dev_loss:9.6f} | '
                f'Best Dev Loss (#{epoch_idx_best}): {dev_loss_best:9.6f}'
            )

            # Check stopping criteria
            if epoch_idx - epoch_idx_best >= FLAGS.num_epoch_after_best:
                break

    def run_test(self):

        self._load_model()
        self._run_eval_epoch(self.test_batcher)
        # self.scorer(y_true, y_pred)
        # self._save_result(result_file, **res)

    ########################################################################################################################
    # Routines for an epoch

    def _run_train_epoch(self, batcher):

        self.model.train()

        total_loss_val = 0.
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(batcher)):

            print(batch_idx, x.shape, y.shape)

            # # Run model and compute loss
            # output, _ = self.model(torch.tensor(
            #     input,
            #     device=self.device,
            #     dtype=torch.float,
            # ))
            # loss = self.model.loss(
            #     input=output,
            #     target=torch.tensor(
            #         label,
            #         device=self.device,
            #         dtype=torch.long,
            #     ),
            #     weight=torch.tensor(
            #         batcher.label_weight,
            #         device=self.device,
            #         dtype=torch.float,
            #     ),
            # )

        return 1

    def _run_eval_epoch(self, batcher):
        """Run evaluation for an epoch.
        """

        self.model.eval()

        total_loss_val = 0.
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(batcher)):

            print(batch_idx, x.shape, y.shape)

        return 0

    ########################################################################################################################
    # Routines model I/O

    def _load_model(self):
        pass

    def _save_model(self):
        pass
