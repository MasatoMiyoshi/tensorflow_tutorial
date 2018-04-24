# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import argparse
import numpy as np
import tensorflow as tf
import train

def create_hparams():
    return tf.contrib.training.HParams(
        # Data
        src='',
        tgt='',
        num_train_steps = 10
    )

def run_main():
    hparams = create_hparams()
    train.train(hparams)

if __name__ == "__main__":
    run_main()
### EOF
