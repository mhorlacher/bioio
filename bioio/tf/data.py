import struct

import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow_datasets.core.features.features_dict import FeaturesDict

import bioio

# %%
def numpy_to_torch(x):
    if isinstance(x, str) or isinstance(x, bytes):
        if isinstance(x, str):
            x = x.encode('UTF-8')
        x = np.frombuffer(x, dtype=np.uint8)
    return torch.Tensor(x)

# %%
def load_index(filepath):
    return np.array(pd.read_csv(filepath, sep='\t', header=None)[0], dtype=np.int64)

# %%
class GFileTFRecord:
    def __init__(self, filepath, features=None, index=None):
        self._gfile_tfrecord = tf.io.gfile.GFile(filepath, 'rb')
        self.features = self._read_features(features) if features else None
        self.index = self._read_index(index) if index else None

    def __call__(self, offset, deserialize=True, validate=False, to_numpy=False, to_torch=False):
        try:
            proto = self._read_proto(offset, validate)
        except:
            raise ValueError(f'Invalid record at offset {offset}.')
        
        if not deserialize:
            return proto
        else:
            return self._deserialize_proto(proto, to_numpy, to_torch)
    
    def __getitem__(self, idx, **kwargs):
        if self.index is None:
            raise ValueError('Index not specified.')
        return self(self.index[idx], **kwargs)
    
    def __iter__(self):
        while True:
            try:
                yield self._deserialize_proto(self._read_next_proto())
            except:
                break
    
    def as_tf_data_iterator(self):
        raise NotImplementedError()
    
    def as_numpy_iterator(self):
        raise NotImplementedError()

    def as_torch_iterator(self):
        raise NotImplementedError()
    
    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            if hasattr(self, '_len'):
                return self._len
            else:
                self._len = self._get_length()
                return self._len
    
    def _get_length(self):
        n = 0
        for _ in iter(self):
            n += 1
        return n
    
    @property
    def size(self):
        return self._gfile_tfrecord.size()

    def _deserialize_proto(self, proto, to_numpy=False, to_torch=False):
        assert not (to_numpy and to_torch), 'Cannot convert to both numpy and torch.'

        if self.features is None:
            raise ValueError('Features not specified.')
        
        example = self.features.deserialize_example(proto)
        if to_numpy:
            example = tf.nest.map_structure(lambda x: x.numpy(), example)
        if to_torch:
            example = tf.nest.map_structure(lambda x: numpy_to_torch(x.numpy()), example)
        return example
    
    def _read_proto(self, offset, validate=False):
        # seek to offset
        self._gfile_tfrecord.seek(offset)
        return self._read_next_proto(validate)
    
    def _read_next_proto(self, validate=False):
        # get proto length
        proto_len_bytes = self._gfile_tfrecord.read(8)
        if len(proto_len_bytes) == 0:
            return None
        proto_len = struct.unpack('q', proto_len_bytes)[0]

        # proto length crc
        proto_len_crc = self._gfile_tfrecord.read(4)
        if validate:
            raise NotImplementedError('CRC validation not implemented.')

        # proto bytes
        proto_bytes = self._gfile_tfrecord.read(proto_len)

        # proto bytes crc
        proto_bytes_crc = self._gfile_tfrecord.read(4)
        if validate:
            raise NotImplementedError('CRC validation not implemented.')
        
        return proto_bytes

    def _read_features(self, features):
        if isinstance(features, FeaturesDict):
            return features
        elif isinstance(features, str):
            return bioio.tf.ops.features_from_json_file(features)
        else:
            raise ValueError(f'Invalid features type: {type(features)}')
        
    def _read_index(self, index):
        if isinstance(index, np.ndarray):
            return index
        elif isinstance(index, str):
            return load_index(index)
        else:
            raise ValueError(f'Invalid features type: {type(index)}')