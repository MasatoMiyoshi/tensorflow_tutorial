# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import time
import collections

import numpy as np
import tensorflow as tf

from .utils import iterator_utils
from .utils import vocab_utils

class TrainModel(
        collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
    pass

def create_train_model(model_creator, hparams):
    src_file = hparams.src_file
    tgt_file = hparams.tgt_file
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container("train"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)

        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)

        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            hparams.sos,
            hparams.eos,
            hparams.random_seed,
            hparams.num_buckets)

        with tf.device("/cpu:0"):
            model = model_creator(
                hparams,
                iterator=iterator,
                mode="train"
                source_vocab_table=src_vocab_table,
                target_vocab_table=tgt_vocab_table)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator)
### EOS
