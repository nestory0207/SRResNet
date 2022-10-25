import os
import pdb
import tensorflow as tf
import glob
from tensorflow.python.data.experimental import AUTOTUNE

def create_train_valid_datasets(dataset_parameters, train_mappings, train_batch_size=16):
    train_dataset = create_train_dataset(dataset_parameters, train_mappings, train_batch_size)
    valid_dataset = create_valid_dataset(dataset_parameters)

    return train_dataset, valid_dataset

def create_train_dataset(dataset_parameters, train_mappings, batch_size):
    hr_dataset = image_dataset_from_directory(dataset_parameters.save_data_directory, "Train/HR")
    lr_dataset = image_dataset_from_directory(dataset_parameters.save_data_directory, dataset_parameters.train_directory)

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    for mapping in train_mappings:
        dataset = dataset.map(mapping, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

def create_valid_dataset(dataset_parameters):
    lr_dataset = image_dataset_from_directory(dataset_parameters.save_data_directory, dataset_parameters.valid_directory)
    hr_dataset = image_dataset_from_directory(dataset_parameters.save_data_directory, "Valid/HR")

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

def image_dataset_from_directory(data_directory, image_directory):
    image_path = os.path.join(data_directory, image_directory)

    fnames = sorted(glob.glob(image_path + "/*.png"))
    dataset = tf.data.Dataset.from_tensor_slices(fnames)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

    cache_directory = os.path.join(data_directory, "cache", image_directory)
    os.makedirs(cache_directory, exist_ok=True)
    cache_file = cache_directory + "/cache"

    dataset = dataset.cache(cache_file)

    if not os.path.exists(cache_file + ".index"):
        populate_cache(dataset, cache_file)

    return dataset

def populate_cache(dataset, cache_file):
    print(f'Begin caching in {cache_file}.')
    for _ in dataset: pass
    print(f'Completed caching in {cache_file}.')