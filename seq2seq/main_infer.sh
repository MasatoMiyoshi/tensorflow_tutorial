#!/bin/bash

# python main.py \
#        --src_vocab_file=./data/vocab_file_os.txt \
#        --tgt_vocab_file=./data/vocab_file_os.txt \
#        --share_vocab=true \
#        --out_dir=./model_20180525_01 \
#        --num_layers=1 \
#        --num_units=128 \
#        --inference_input_file=./data/faq_train_in.txt \
#        --inference_output_file=./model_20180525_01/output_infer.txt

python main.py \
       --src_vocab_file=./data/pretrained_embed_file/vocab_file_os.txt \
       --tgt_vocab_file=./data/pretrained_embed_file/vocab_file_os.txt \
       --share_vocab=true \
       --out_dir=./model_20180528_01_pretrained_emb \
       --num_layers=1 \
       --num_units=128 \
       --inference_input_file=./data/faq_train_in.txt \
       --inference_output_file=./model_20180528_01_pretrained_emb/output_infer.txt
### EOF
