import json
import time
import tensorflow as tf
from tensorflow.keras import models

from models import CNN_Encoder, RNN_Decoder

TOP_K = 5000

def load_image_InceptionV3(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def build_features_extract_InceptionV3():
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    return tf.keras.Model(new_input, hidden_layer)


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


def load_latest_imgcap(checkpoint_path):
    embedding_dim = 256
    units = 512
    vocab_size = TOP_K + 1

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_man = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    ckpt.restore(ckpt_man.latest_checkpoint)

    return encoder, decoder


def formatt_result(result: list):
    result.remove('<end>')
    result.append('.')
    return ' '.join(result)


def inference(image, models, random_seed=None):
    feature_extractor, tokenizer, max_length, encoder, decoder = models

    hidden = decoder.reset_state(batch_size=1)
    img_batch = tf.expand_dims(load_image_InceptionV3(image), 0)
    img_batch = feature_extractor(img_batch)
    img_batch = tf.reshape(img_batch, (img_batch.shape[0], -1, img_batch.shape[3]))

    features = encoder(img_batch)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    result = []
    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        predicted_id = None
        if random_seed:
            predicted_id = tf.random.categorical(predictions, 1, seed=random_seed)[0][0].numpy()
        else:
            predicted_id = tf.argmax(predictions, 1)[0].numpy()
        
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return formatt_result(result)

        dec_input = tf.expand_dims([predicted_id], 0)

    return formatt_result(result)


if '__main__' == __name__:
    image_file = 'surf.jpg'

    annotation_file='./annotations/captions_train2014.json' 
    checkpoint_path='./checkpoints/train/'

    ts = time.time()
    feature_extractor = build_features_extract_InceptionV3()
    tokenizer, max_length = load_tokenizer(annotation_file)
    encoder, decoder = load_latest_imgcap(checkpoint_path)
    te = time.time()

    load_model_time = te - ts

    models = [feature_extractor, 
                tokenizer, max_length, 
                encoder, decoder]

    ts = time.time()
    print(inference(image_file, models))
    te = time.time()

    inference_time = te - ts
    print(f'Loading models takes {load_model_time} seconds')
    print(f'Inference takes {inference_time} seconds')
