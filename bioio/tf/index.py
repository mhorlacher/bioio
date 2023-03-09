# %%
import struct

import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf

from .ops import features_from_json_file

# %%
def index_tfrecord(tfrecord, index_filepath, pbar=None, proto_fn=None):
    with open(tfrecord, 'rb') as tfr, open(index_filepath, 'w') as idx:
        pbar = (tqdm.tqdm() if pbar is not None else None)
        while True:
            current = tfr.tell()
            try:
                # byte length
                byte_len = tfr.read(8)
                if len(byte_len) == 0:
                    break

                # crc length
                byte_len_crc = tfr.read(4)
                proto_len = struct.unpack('q', byte_len)[0]

                # proto
                proto = tfr.read(proto_len)

                # crc
                tfr.read(4)
                print(str(current) + '\t' + str(tfr.tell() - current) + ('\t' + str(proto_fn(proto)) if proto_fn is not None else ''), file=idx)
            except Exception:
                print('Not a valid TFRecord file.')
                break
            
            if pbar is not None:
                pbar.update(1)

# %%
def load_index(filepath):
    return np.array(pd.read_csv(filepath, sep='\t', header=None), dtype=np.int64)

# %%
def index_to_dataset(filepath):
    index = load_index(filepath)
    dataset = tf.data.Dataset.from_tensor_slices((index[:, 0], index[:, 1]))
    return dataset

# %%
class GFileTFRecord:
    def __init__(self, filepath, features=None):
        self._gfile_tfrecord = tf.io.gfile.GFile(filepath, 'rb')
        self.features = features
    
    def _seek_and_read(self, offset, length):
        offset, length = int(offset), int(length)

        self._gfile_tfrecord.seek(offset)
        return self._gfile_tfrecord.read(length)[(8+4):-4]

    def read_proto(self, offset, length):
        proto_bytes = tf.py_function(self._seek_and_read, inp=[offset, length], Tout=tf.string)
        proto_bytes.set_shape(shape=())
        return proto_bytes
    
    def read_example(self, offset, length):
        if self.features is None:
            raise ValueError('Features are required for deserialization.')

        proto = self.read_proto(offset, length)
        return self.features.deserialize_example(proto)

# %%
# def read_indexed_tfrecord(tfrecord_file, index_file, features_file, shuffle=False):
#     features = features_from_json_file(features_file)
#     gfile_tfrecord = GFileTFRecord(tfrecord_file, features)
#     dataset = index_to_dataset(index_file)
#     if shuffle:
#         dataset = dataset.shuffle(shuffle)
#     dataset = dataset.map(gfile_tfrecord.read_example)
#     return dataset

# %%
def load_indexed_tfrecord(tfrecord_file, features_file=None, index_file=None, shuffle=False): # TODO: Implement caching!
    if features_file is None:
        features_file = tfrecord_file + '.features.json'
    features = features_from_json_file(features_file)

    if index_file is None:
        index_file = tfrecord_file + '.idx'
    dataset = index_to_dataset(index_file)
    print(dataset.cardinality())

    gfile_tfrecord = GFileTFRecord(tfrecord_file)

    # shuffle indices
    if shuffle:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # load proto
    dataset = dataset.map(gfile_tfrecord.read_proto)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # deserialize proto to example
    dataset = dataset.map(features.deserialize_example)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
