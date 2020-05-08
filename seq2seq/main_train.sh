#!/bin/bash

# python main.py \
#        --src_file=./data/faq_train_in.txt \
#        --tgt_file=./data/faq_train_out.txt \
#        --src_vocab_file=./data/vocab_file_os.txt \
#        --tgt_vocab_file=./data/vocab_file_os.txt \
#        --src_max_len=50 \
#        --tgt_max_len=50 \
#        --share_vocab=true \
#        --out_dir=./model_20180525_01 \
#        --num_train_steps=2400 \
#        --steps_per_stats=50 \
#        --num_layers=1 \
#        --num_units=128 \
#        --dropout=0.2

python main.py \
       --src_file=./data/faq_train_in.txt \
       --tgt_file=./data/faq_train_out.txt \
       --src_vocab_file=./data/pretrained_embed_file/vocab_file_os.txt \
       --tgt_vocab_file=./data/pretrained_embed_file/vocab_file_os.txt \
       --src_embed_file=./data/pretrained_embed_file/word2vec_os.txt \
       --tgt_embed_file=./data/pretrained_embed_file/word2vec_os.txt \
       --src_max_len=50 \
       --tgt_max_len=50 \
       --share_vocab=true \
       --out_dir=./model_20180528_01_pretrained_emb \
       --num_train_steps=2400 \
       --steps_per_stats=50 \
       --num_layers=1 \
       --num_units=128 \
       --dropout=0.2
### EOF
