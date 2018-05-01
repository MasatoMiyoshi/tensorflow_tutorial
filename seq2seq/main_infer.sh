#!/bin/bash

python main.py \
       --src_vocab_file=./data/vocab_file.txt \
       --tgt_vocab_file=./data/vocab_file.txt \
       --share_vocab=true \
       --out_dir=./model_20180501_03 \
       --num_units=128 \
       --inference_input_file=./data/faq_title_mrphs_fix.txt \
       --inference_output_file=./model_20180501_03/output_infer
### EOF
