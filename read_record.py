# Name: Cory Nezin
# Date: 09/08/2017 (Modified 05/24/2018)
# Goal: Create an iterator for a given TFRecords file

import os
import tensorflow as tf

TFRECORDS_DIRECTORY = "reward_records"
from constants import *

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

feature = {'state_raw': tf.FixedLenFeature([], tf.string),
           'reward': tf.FixedLenFeature([], tf.int64)}

def get_dataset():
    filename = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filename)
    return filename, dataset

def get_iterator(batch_size=64, shuffle=False, buffer_size=None):
    if shuffle:
        assert buffer_size is not None

    filename, dataset = get_dataset()

    # augment data
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(parse)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    return filename, iterator

def get_oneshot_iterator(filename, preproc=None):
    dataset = tf.data.TFRecordDataset(os.path.join(TFRECORDS_DIRECTORY, filename))
    dataset = dataset.map(parse)

    iterator = dataset.make_one_shot_iterator()
    return iterator

def parse(b):
    features = tf.parse_single_example(b, features=feature)
    state_flat = tf.decode_raw(features['state_raw'], tf.float32)  # actual data
    label = tf.cast(features['reward'], tf.int64)
    label = tf.one_hot(label, n_classes)
    # Reshape binary vector into an image
    state = tf.reshape(state_flat, [10,10])
    return ({'label': label, 'state': state})

