import tensorflow as tf
import collections
import random
import numpy as np
import os
import time
import json

from PIL import Image
from tqdm import tqdm
from model import BahdanauAttention, CNN_Encoder, RNN_Decoder

##########################
#> Load MS-COCO dataset <#
##########################
annotation_file = 'annotations/captions_train2014.json'
image_folder = '/train2014/'
PATH = os.path.abspath('.') + image_folder

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Group all captions together having the same image ID.
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    caption = f"<start> {val['caption']} <end>"
    image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
    image_path_to_caption[image_path].append(caption)

train_image_paths = list(image_path_to_caption.keys())
random.seed(42)
random.shuffle(train_image_paths)

train_captions = []
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))

##########################################
#> Preprocess and tokenize the captions <#
##########################################
# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)

###########################
#> Prepare training data <#
###########################
img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)

# Create training and validation sets using an 80-20 split randomly.
img_keys = list(img_to_cap_vector.keys())
random.seed(42)
random.shuffle(img_keys)

img_name_train = []
cap_train = []
for img_key in img_keys:
    capt_len = len(img_to_cap_vector[img_key])
    img_name_train.extend([img_key] * capt_len)
    cap_train.extend(img_to_cap_vector[img_key])

#########################################
#> Create tf.data dataset for training <#
#########################################
# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

###########
#> Model <#
###########

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

################
#> Checkpoint <#
################
checkpoint_path = "./checkpoints/train/"
ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 1
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) * 5 + 1
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

##############
#> Training <#
##############
@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

EPOCHS = 35

with open('log.txt', 'a') as f:
    start_info = f'Number of epochs {EPOCHS} Number of steps per epoch {num_steps}\n'
    f.write(start_info)
    print(start_info)

    for epoch in range(start_epoch, EPOCHS+1):
        start_epoch = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in tqdm(enumerate(dataset), position=0, leave=True):
            start_step = time.time()
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            average_batch_loss = batch_loss.numpy()/int(target.shape[1])

            batch_info = f'Epoch {epoch} Batch {batch}/{num_steps} Loss {average_batch_loss:.4f} Time {time.time()-start_step:.2f} sec\n' 
            f.write(batch_info)
            print(batch_info)

        if epoch % 5 == 0:
            ckpt_manager.save()
        
        epoch_loss_info = f'Epoch {epoch} Loss {total_loss/num_steps:.6f}\n'
        f.write(epoch_loss_info)
        print(epoch_loss_info)

        epoch_time_info = f'Time taken for 1 epoch {time.time()-start_epoch:.2f} sec\n\n'
        f.write(epoch_time_info)
        print(epoch_time_info)