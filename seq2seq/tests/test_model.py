# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import model
from utils import common_test_utils

class TestModel(tf.test.TestCase):
    def test_init(self):
        hparams = common_test_utils.create_test_hparams()
        iterator, src_vocab_table, tgt_vocab_table = common_test_utils.create_test_iterator(hparams, tf.estimator.ModeKeys.TRAIN)
        model_creator = model.Model
        train_model = model_creator(hparams, tf.estimator.ModeKeys.TRAIN, iterator, src_vocab_table, tgt_vocab_table)

if __name__ == "__main__":
    tf.test.main()
### EOF
