import json
import tensorflow as tf

TOP_K = 5000

def prepare_caption(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    captions = [f'<start> {val["caption"]} <end>' for val in annotations['annotations']]
    return captions


def load_tokenizer(annotation_file):
    captions = prepare_caption(annotation_file)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_K, oov_token="<unk>", filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)

    tokenizer.word_index['<pads>'] = 0
    tokenizer.index_word[0] = '<pads>'

    max_length = max(len(t) for t in tokenizer.texts_to_sequences(captions))
    return tokenizer, max_length