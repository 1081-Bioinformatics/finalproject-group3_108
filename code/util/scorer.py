#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Compute scores for model outputs"""

__author__ = 'Jia-Yu Lu <jeanie0807@gmail.com>'

from absl import flags
from absl import logging

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

################################################################################################################################

FLAGS = flags.FLAGS

################################################################################################################################


class Scorer:

    # Compute AUC score
    def auc(self, *, y_true, y_score):
        return metrics.roc_auc_score(y_true, y_score)

    # Compute AUC score and CI
    def __call__(self, *, y_true, y_score):

        # Bootstrap
        num_bootstrap = 1000
        bootstrap_scores = []

        for i in range(num_bootstrap):
            # bootstrap by sampling with replacement on the prediction indices
            indices = np.random.randint(0, len(y_score), len(y_score))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = metrics.roc_auc_score(y_true[indices], y_score[indices])
            bootstrap_scores.append(score)

        sorted_scores = np.array(bootstrap_scores)
        sorted_scores.sort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

        msg = f"AUC {metrics.roc_auc_score(y_true, y_score):.2f}" + \
            f" (95%CI {confidence_lower:.2f} - {confidence_upper:.2f})"

        print(msg)

        self.draw(
            y_true=y_true,
            y_score=y_score,
            msg=msg,
        )

    ################################################################################################################################

    # Draw ROC curve
    def draw(self, *, y_true, y_score, msg):

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.suptitle(FLAGS.full_name)
        plt.title(msg)

        file = FLAGS.figure_file
        os.makedirs(os.path.dirname(file), exist_ok=True)
        logging.info(f'Saving figure into {file}')
        plt.savefig(file)
