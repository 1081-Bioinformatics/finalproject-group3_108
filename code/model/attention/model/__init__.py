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

    def __init__(self, *, num_feature):

        super().__init__()

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=1,
            num_heads=1,
        )

        self.linear = torch.nn.Linear(
            in_features=num_feature,
            out_features=1,
        )

        self.loss = torch.nn.functional.binary_cross_entropy

    def forward(self,
                x,  # [B, F,]
                ):

        x = x.t().unsqueeze(-1)  # [F, B, 1]

        (
            attn_output,  # [F, B, 1]
            _,
        ) = self.attn(
            query=x,  # [F, B, 1]
            key=x,    # [F, B, 1]
            value=x,  # [F, B, 1]
        )

        attn_output = attn_output.squeeze(-1).t()  # [B, F]
        logit = self.linear(attn_output).squeeze(-1)  # [B]
        score = torch.sigmoid(logit)  # [B]
        return score
