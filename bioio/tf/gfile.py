import struct

import tensorflow as tf
import numpy as np
from tensorflow_datasets.core.features.features_dict import FeaturesDict

import bioio

# %%
class GFileTFRecord:
    def __init__(self, filepath, features=None, index=None):
        self._gfile_tfrecord = tf.io.gfile.GFile(filepath, 'rb')
        self.features = self._read_features(features) if features is not None else None
        self.index = self._read_index(index) if index is not None else None

    def __call__(self, offset, deserialize=True, to_numpy=False, to_torch=False, validate=False):
        try:
            proto = self._read_proto(offset, validate)
        except:
            raise ValueError(f'Invalid record at offset {offset}.')
        
        if (self.features is None) or (not deserialize):
            return proto
        else:
            return self.deserialize(proto, to_numpy, to_torch)
    
    def __getitem__(self, idx, **kwargs):
        if self.index is None:
            raise ValueError('Index not specified.')
        return self(self.index[idx], **kwargs)
    
    def __iter__(self):
        while True:
            try:
                proto = self._read_next_proto()
                if self.features is None:
                    yield proto
                else:
                    yield self.deserialize(proto)
            except:
                break
    
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

    def as_tf_data_iterator(self, shuffle=False):
        raise NotImplementedError()
    
    def as_numpy_iterator(self, shuffle=False):
        raise NotImplementedError()

    def as_torch_iterator(self, shuffle=False):
        raise NotImplementedError()

    def deserialize(self, proto, to_numpy=False, to_torch=False):
        assert not (to_numpy and to_torch), 'Cannot convert to both numpy and torch.'

        if self.features is None:
            raise ValueError('Features not specified.')
        
        example = self.features.deserialize_example(proto)
        if to_numpy:
            example = tf.nest.map_structure(lambda x: x.numpy(), example)
        if to_torch:
            example = tf.nest.map_structure(lambda x: bioio.torch.utils.numpy_to_torch(x.numpy()), example)
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
            return bioio.tf.utils.features_from_json_file(features)
        else:
            raise ValueError(f'Invalid features type: {type(features)}')
        
    def _read_index(self, index):
        if isinstance(index, np.ndarray):
            return index
        elif isinstance(index, str):
            return bioio.tf.index.load_index(index)
        else:
            raise ValueError(f'Invalid features type: {type(index)}')

# %%