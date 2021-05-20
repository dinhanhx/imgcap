import tensorflow as tf
import time

from tqdm import tqdm
from model import CNN_Encoder, RNN_Decoder
from dataset import e2e_create_tf_dataset
from tokenization import TOP_K, load_tokenizer

annotation_file = 'annotations/captions_train2014.json'
image_folder = 'train2014/'

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = TOP_K + 1
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
# features_shape = 2048
# attention_features_shape = 64

tokenizer, _ = load_tokenizer(annotation_file)
dataset, dataset_size = e2e_create_tf_dataset(annotation_file, image_folder, BATCH_SIZE, BUFFER_SIZE)
num_steps = dataset_size // BATCH_SIZE

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

EPOCHS = 40

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