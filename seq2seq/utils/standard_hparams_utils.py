# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import

import tensorflow as tf

def create_standard_hparams():
    return tf.contrib.training.HParams(
        # Data
        src_file="",
        tgt_file="",
        src_vocab_file="",
        tgt_vocab_file="",
        out_dir="",
        # Networks
        num_units=512,
        num_layers=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.2,
        unit_type="lstm",
        time_major=True,
        # Train
        optimizer="sgd",
        batch_size=128,
        init_op="uniform",
        init_weight=0.1,
        max_gradient_norm=5.0,
        learning_rate=1.0,
        warmup_steps=0,
        warmup_scheme="t2t",
        decay_scheme="luong234",
        colocate_gradients_with_ops=True,
        num_train_steps=12000,
        # Data constraints
        num_buckets=5,
        max_train=0,
        src_max_len=50,
        tgt_max_len=50,
        src_max_len_infer=0,
        tgt_max_len_infer=0,
        # Data format
        sos="<s>",
        eos="</s>",
        subword_option="",
        check_special_token=True,
        # Misc
        forget_bias=1.0,
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=100,
        share_vocab=False,
        log_device_placement=False,
        random_seed=None,
        beam_width=0,
        length_penalty_weight=0.0,
        # For inference
        infer_batch_size=32,
        sampling_temperature=0.0,
        num_translations_per_input=1,
    )
### EOF
