# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import time

import numpy as np
import tensorflow as tf

import model
import model_helper
from utils import misc_utils as utils

def load_data(inference_input_file, hparams=None):
    """Load inference data."""
    with codecs.getreader("utf-8")(
            tf.gfile.GFile(inference_input_file, mode="rb")) as f:
        inference_data = f.read().splitlines()

    return inference_data

def inference(ckpt,
              inference_input_file,
              inference_output_file,
              hparams,
              num_workers=1):
    """Perform inference."""
    model_creator = model.Model
    infer_model = model_helper.create_infer_model(model_creator, hparams)

    single_worker_inference(
        infer_model,
        ckpt,
        inference_input_file,
        inference_output_file,
        hparams)

def single_worker_inference(infer_model,
                            ckpt,
                            inference_input_file,
                            inference_output_file,
                            hparams):
    """Inference with a single worker."""
    output_infer = inference_output_file

    # Read data
    infer_data = load_data(inference_input_file, hparams)

    with tf.Session(config=utils.get_config_proto(), graph=infer_model.graph) as sess:
        loaded_infer_model = model_helper.load_model(infer_model.model, ckpt, sess, "infer")
        sess.run(infer_model.iterator.initializer,
                 feed_dict={
                     infer_model.src_placeholder: infer_data,
                     infer_model.batch_size_placeholder: hparams.infer_batch_size
                 })
        # Decode
        utils.print_out("# Start decoding")
        _decode_and_evaluate("infer",
                             loaded_infer_model,
                             sess,
                             output_infer,
                             ref_file=None,
                             subword_option=None,
                             beam_width=hparams.beam_width,
                             tgt_eos=hparams.eos,
                             num_translations_per_input=hparams.num_translations_per_input)

def _decode_and_evaluate(name,
                         model,
                         sess,
                         trans_file,
                         ref_file,
                         subword_option,
                         beam_width,
                         tgt_eos,
                         num_translations_per_input=1,
                         decode=True):
    """Decode a test set and compute a score according to the evaluation task."""
    # Decode
    if decode:
        utils.print_out("  decoding to output %s." % trans_file)

        start_time = time.time()
        num_sentences = 0
        with codecs.getwriter("utf-8")(tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
            trans_f.write("")  # Write empty string to ensure file is created.

            num_translations_per_input = max(min(num_translations_per_input, beam_width), 1)
            while True:
                try:
                    outputs, _ = model.decode(sess)
                    if beam_width == 0:
                        outputs = np.expand_dims(outputs, 0)

                    batch_size = outputs.shape[1]
                    num_sentences += batch_size

                    for sent_id in range(batch_size):
                        for beam_id in range(num_translations_per_input):
                            translation = _get_translation(
                                outputs[beam_id],
                                sent_id,
                                tgt_eos=tgt_eos,
                                subword_option=subword_option)
                            trans_f.write((translation + b"\n").decode("utf-8"))
                except tf.errors.OutOfRangeError:
                    break
    return

def _get_translation(outputs, sent_id, tgt_eos, subword_option):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
    # Select a sentence
    output = outputs[sent_id, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]

    translation = utils.format_text(output)
    return translation
### EOF
