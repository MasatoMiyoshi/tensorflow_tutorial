# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import time
import argparse
import numpy as np
import tensorflow as tf
# import model

def train(hparams):
    num_train_steps = hparams.num_train_steps
    # model_creator = model.Model

    train_model = model_helper.create_train_model(model_creator, hparams)

    global_step = train_model.global_step.eval(session=train_sess)
    while global_step < num_train_steps:
        ### Run a step ###
        start_time = time.time()
        break
### EOS
