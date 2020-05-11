import os
import tensorflow as tf
import tensorflow_datasets as tfds
import constant as cst

def get_toknizer(train_examples=None):
    tokenizer_en_filename = 'tokenizer_en'
    tokenizer_pt_filename = 'tokenizer_pt'
    suffix = '.subwords'

    if train_examples is None or \
       (os.path.exists(tokenizer_en_filename + suffix) and \
        os.path.exists(tokenizer_pt_filename + suffix)):
        tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(
            tokenizer_en_filename)
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file(
            tokenizer_pt_filename)
    else:
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

        tokenizer_en.save_to_file(tokenizer_en_filename)
        tokenizer_pt.save_to_file(tokenizer_pt_filename)

    return tokenizer_en, tokenizer_pt
def get_dataset():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en, tokenizer_pt = get_toknizer(train_examples=train_examples)

    sample_string = 'Transformer is awesome.'

    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    assert original_string == sample_string

    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
            lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            lang2.numpy()) + [tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en

    def filter_max_length(x, y, max_length=cst.MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    train_preprocessed = (
        train_examples
        .map(tf_encode)
        .filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        .cache()
        .shuffle(cst.BUFFER_SIZE))

    val_preprocessed = (
        val_examples
        .map(tf_encode)
        .filter(filter_max_length))

    train_dataset = (train_preprocessed
                     .padded_batch(cst.BATCH_SIZE, padded_shapes=([None], [None]))
                     .prefetch(tf.data.experimental.AUTOTUNE))

    val_dataset = (val_preprocessed
                   .padded_batch(cst.BATCH_SIZE, padded_shapes=([None], [None])))

    return train_dataset, val_dataset

if __name__ == '__main__':
    train_dataset, val_dataset = get_dataset()

    pt_batch, en_batch = next(iter(val_dataset))
    print(pt_batch, en_batch)
