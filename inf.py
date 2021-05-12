import time
import tensorflow as tf

from model import CNN_Encoder, RNN_Decoder, FeatureExtraction
from tokenization import load_tokenizer, TOP_K

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
    img_batch = tf.expand_dims(FeatureExtraction.load_image_InceptionV3(image), 0)
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
    feature_extractor = FeatureExtraction.build_model_InceptionV3()
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
