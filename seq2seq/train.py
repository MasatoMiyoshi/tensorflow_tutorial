# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import time

import numpy as np
import tensorflow as tf

import model
from .utils import misc_utils as utils

def train(hparams):
    """Train a translation model."""
    out_dir = hparams.out_dir
    num_train_steps = hparams.num_train_steps

    model_creator = model.Model

    train_model = model_helper.create_train_model(model_creator, hparams)

    summary_name = "train_log"
    model_dir = hparams.out_dir

    # Log and output files
    log_file = os.path.join(out_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    utils.print_out("# log_files=%s" % log_file, log_f)

    # TensorFlow model
    config_proto = utils.get_config_proto(
        log_device_placement=log_device_placement,
        num_intra_threads=hparams.num_intra_threads,
        num_inter_threads=hparams.num_inter_threads)
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)

    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_model.model, model_dir, train_sess, "train")

    # Summary writer
    summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, summary_name), train_model.graph)

    last_stats_step = global_step

    # This is the training loop.
    stats, info, start_train_time = before_train(
        loaded_train_model, train_model, train_sess, global_step, hparams, log_f)
    while global_step < num_train_steps:
        ### Run a step ###
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
            hparams.epoch_step += 1
        except tf.errors.OutOfRangeError:
            # Finished going through the training dataset.  Go to next epoch.
            hparams.epoch_step = 0

            train_sess.run(
                train_model.iterator.initializer,
                feed_dict={train_model.skip_count_placeholder: 0})
            continue

        # Process step_result, accumulate stats, and write summary
        global_step, info["learning_rate"], step_summary = update_stats(
            stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)
        break

    # Done training
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(out_dir, "seq2seq.ckpt"),
        global_step=global_step)

    summary_writer.close()

def before_train(loaded_train_model, train_model, train_sess, global_step, hparams, log_f):
    """Misc tasks to do before training."""
    stats = init_stats()
    info = {"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
            "avg_grad_norm": 0.0,
            "learning_rate": loaded_train_model.learning_rate.eval(session=train_sess)}
    start_train_time = time.time()
    utils.print_out("# Start step %d, lr %g, %s" %
                    (global_step, info["learning_rate"], time.ctime()), log_f)

    # Initialize all of the iterators
    skip_count = hparams.batch_size * hparams.epoch_step
    utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
    train_sess.run(train_model.iterator.initializer,
                   feed_dict={train_model.skip_count_placeholder: skip_count})

    return stats, info, start_train_time

def init_stats():
    """Initialize statistics that we want to accumulate."""
    return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
            "total_count": 0.0, "grad_norm": 0.0}

def update_stats(stats, start_time, step_result):
    """Update stats: write summary and accumulate statistics."""
    (_, step_loss, step_predict_count, step_summary, global_step,
     step_word_count, batch_size, grad_norm, learning_rate) = step_result

    # Update statistics
    stats["step_time"] += (time.time() - start_time)
    stats["loss"] += (step_loss * batch_size)
    stats["predict_count"] += step_predict_count
    stats["total_count"] += float(step_word_count)
    stats["grad_norm"] += grad_norm

    return global_step, learning_rate, step_summary
### EOS
