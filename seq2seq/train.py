# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import time
import math

import numpy as np
import tensorflow as tf

import model
import model_helper
from utils import misc_utils as utils

def train(hparams):
    """Train a seq2seq model."""
    log_device_placement = hparams.log_device_placement
    out_dir = hparams.out_dir
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats

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
            utils.print_out("# Finished an epoch, step %d." % global_step)

            train_sess.run(
                train_model.iterator.initializer,
                feed_dict={train_model.skip_count_placeholder: 0})
            continue

        # Process step_result, accumulate stats, and write summary
        global_step, info["learning_rate"], step_summary = update_stats(
            stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)

        # Once in a while, we print statistics.
        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step
            is_overflow = process_stats(stats, info, global_step, steps_per_stats, log_f)
            print_step_info("  ", global_step, info, log_f)

            if is_overflow:
                break

            # Reset statistics
            stats = init_stats()

    # Done training
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(out_dir, "seq2seq.ckpt"),
        global_step=global_step)

    summary_writer.close()

    return global_step

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

def print_step_info(prefix, global_step, info, log_f):
    """Print all info at the current global step."""
    utils.print_out(
        "%sstep %d lr %g step-time %.2fs wps %2.fK ppl %.2f gN %.2f, %s" %
        (prefix, global_step, info["learning_rate"], info["avg_step_time"],
         info["speed"], info["train_ppl"], info["avg_grad_norm"],
         time.ctime()),
        log_f)

def process_stats(stats, info, global_step, steps_per_stats, log_f):
    """Update info and check for overflow."""
    # Update info
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
    info["train_ppl"] = utils.safe_exp(stats["loss"] / stats["predict_count"])
    info["speed"] = stats["total_count"] / (1000 * stats["step_time"])

    # Check for overflow
    is_overflow = False
    train_ppl = info["train_ppl"]
    if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
        utils.print_out("  step %d overflow, stop early" % global_step, log_f)
        is_overflow = True

    return is_overflow
### EOS
