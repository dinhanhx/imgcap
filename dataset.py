import json
import collections
import random
import tensorflow as tf
import numpy as np

from tokenization import load_tokenizer


def prepare_COCO_dataset(annotation_file, image_folder, seed=42):

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = image_folder + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)

    train_image_paths = list(image_path_to_caption.keys())
    random.seed(seed)
    random.shuffle(train_image_paths)

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    return train_captions, img_name_vector


def prepare_cap_vector(annotation_file, train_captions):
    tokenizer, _ = load_tokenizer(annotation_file)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    return cap_vector


def prepare_training_data(img_name_vector, cap_vector, seed=42):
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    img_keys = list(img_to_cap_vector.keys())
    random.seed(seed)
    random.shuffle(img_keys)

    img_name_train = []
    cap_train = []
    for img_key in img_keys:
        capt_len = len(img_to_cap_vector[img_key])
        img_name_train.extend([img_key] * capt_len)
        cap_train.extend(img_to_cap_vector[img_key])

    return img_name_train, cap_train


def create_tf_dataset(img_name_train, cap_train, batch_size = 64, buffer_size = 1000, seed = 42):
    def map_func(img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
                map_func, [item1, item2], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size, seed).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, len(img_name_train)


def e2e_create_tf_dataset(annotation_file, image_folder, batch_size = 64, buffer_size = 1000, seed = 42):
    train_captions, img_name_vector = prepare_COCO_dataset(annotation_file, image_folder, seed)
    cap_vector = prepare_cap_vector(annotation_file, train_captions)
    img_name_train, cap_train = prepare_training_data(img_name_vector, cap_vector, seed)
    dataset, dataset_size = create_tf_dataset(img_name_train, cap_train, batch_size, buffer_size, seed)
    return dataset, dataset_size


if '__main__' == __name__:
    annotation_file = 'annotations/captions_train2014.json'
    image_folder = 'train2014/'

    # You can write test code or trial code here
