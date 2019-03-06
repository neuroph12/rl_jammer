import numpy as np,os,pickle,tensorflow as tf

from constants import PKL_DIRECTORY, TFRECORDS_DIRECTORY
from data_loader import LoadModRecData

data_directory = PKL_DIRECTORY
output_directory = TFRECORDS_DIRECTORY

def _int64_feature(value):
    '''returns dataset features of type int for classification training'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    '''returns dataset features of type float for inference'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_pkl_to_record(pkl_file, record_name):
    writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, record_name))
    data = LoadModRecData(os.path.join(data_directory, pkl_file), .97, .03, 0)
    batches = data.batch_iter(data.test_idx, 1, 1, use_shuffle=True)
    for batch in batches:
        x, y, s = zip(*batch)
        label = int(np.where(y)[1])
        state = np.asarray(x)
        feature = {'train/label': _int64_feature(label),
                   'train/iq': _bytes_feature(tf.compat.as_bytes(state.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

def write_test_set_to_record(pkl_file, record_name):
    writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, record_name))
    with open(os.path.join(data_directory, pkl_file), 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    for key in data.keys():
        feature = {'train/label': _int64_feature(0),
                   'train/iq': _bytes_feature(tf.compat.as_bytes(np.float32(data[key]).tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

def write_records(ratios):
    data = LoadModRecData(os.path.join(data_directory, ))
    if ratios[0] > 0:
        train_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'train.tfrecord'))
    if ratios[1] > 0:
        val_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'val.tfrecord'))
    if ratios[2] > 0:
        test_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'test.tfrecord'))

def write_split(data_directory, output_directory, train=True):
    
    # if train:
    #     train, val, test = (1.0, 0.0, 0.0)
    #     train_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'train.tfrecord'))
    #     val_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'tmp.tfrecord'))
    # else:
    #     train, val, test = (0.0, 1.0, 0.0)
    #     train_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'tmp.tfrecord'))
    #     val_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'val.tfrecord'))
    
    train, val, test = (.8, .2, 0.0)
    train_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'train.tfrecord'))
    val_writer = tf.python_io.TFRecordWriter(os.path.join(output_directory, 'val.tfrecord'))  

    file_number = 0

    for file_name in os.listdir(data_directory):
        print(file_name)
        if file_name.startswith('.'):
            continue
        file_number = file_number + 1
        print(data_directory + file_name)
        data = LoadModRecData(os.path.join(data_directory, file_name), train, val, test)

        train_batch_size = 1
        train_batches = \
            data.batch_iter(data.train_idx, train_batch_size, 1, use_shuffle=True)
        test_batches = \
            data.batch_iter(data.test_idx, train_batch_size, 1, use_shuffle=True)
        val_batches = \
            data.batch_iter(data.val_idx, train_batch_size, 1, use_shuffle=True)

        for batch in train_batches:
            x, y, s = zip(*batch)
            label = int(np.where(y)[1])
            state = np.asarray(x, dtype=np.float32)
            feature = {'reward':_int64_feature(label),
                       'state_raw':_bytes_feature(tf.compat.as_bytes(state.tostring()))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            train_writer.write(example.SerializeToString())

        for batch in val_batches:
            x, y, s = zip(*batch)
            label = int(np.where(y)[1])
            state = np.asarray(x, dtype=np.float32)
            feature = {'reward':_int64_feature(label),
                       'state_raw':_bytes_feature(tf.compat.as_bytes(state.tostring()))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            val_writer.write(example.SerializeToString())
