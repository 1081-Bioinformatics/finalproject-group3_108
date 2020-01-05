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
                device=self.device,
            )

        self.scorer = Scorer()

        self.model = Model(
            num_feature=self.train_batcher.num_feature,
        )
        self.model.to(device=self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )

    ################################################################

    def run_train(self):

        epoch_idx_best = 0
        dev_loss_best = np.inf

        # Loop for epoch
        for epoch_idx in range(1, FLAGS.num_epoch + 1):

            # Run training and validation
            train_loss = self._run_train_epoch(self.train_batcher)
            dev_loss, _ = self._run_eval_epoch(
                self.dev_batcher)

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
        test_loss, (y_true, y_score,) = self._run_eval_epoch(self.test_batcher)
        logging.info(f'Test Loss: {test_loss:9.6f}')

        self.scorer(
            y_true=y_true,
            y_score=y_score,
        )

    ########################################################################################################################
    # Routines for an epoch

    def _run_train_epoch(self, batcher):

        self.model.train()

        total_loss_val = 0.
        for batch_idx, (x, y_true) in enumerate(tqdm.tqdm(batcher)):

            # Run model and compute loss
            y_score = self.model(x)
            loss = self.model.loss(
                y_score,
                y_true,
            )

            # Add loss
            loss_val = loss.item()
            total_loss_val += loss_val

            # Run backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss_val / batcher.num_sample

    def _run_eval_epoch(self, batcher):
        """Run evaluation for an epoch.
        """

        self.model.eval()

        total_loss_val = 0.
        all_y_true = []
        all_y_score = []
        for batch_idx, (x, y_true) in enumerate(tqdm.tqdm(batcher)):

            # Run model and compute loss
            y_score = self.model(x)
            loss = self.model.loss(
                y_score,
                y_true,
            )
            all_y_true.append(y_true.cpu().data.numpy())
            all_y_score.append(y_score.cpu().data.numpy())

            # Add loss
            loss_val = loss.item()
            total_loss_val += loss_val

        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_score = np.concatenate(all_y_score, axis=0)

        return total_loss_val / batcher.num_sample, (all_y_true, all_y_score,)

    ########################################################################################################################
    # Routines model I/O

    def _load_model(self):
        """Load pretrained model.
        """
        file = FLAGS.model_file
        logging.info(f'Loading model from {file} ...')
        state_dict = torch.load(
            file, map_location=lambda storage, loc: storage)

        self.model.load_state_dict(state_dict)

    def _save_model(self):

        file = FLAGS.model_file
        logging.info(f'Saving model into {file} ...')
        os.makedirs(os.path.dirname(file), exist_ok=True)
        torch.save(self.model.state_dict(), file)
