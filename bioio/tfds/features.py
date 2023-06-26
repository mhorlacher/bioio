# %%
import tensorflow as tf
import tensorflow_datasets as tfds

# %%
class SparseTensor(tfds.features.FeatureConnector):
    def __init__(self, dtype, **kwargs):
        self.dtype = dtype
        self._doc = tfds.features.Documentation()

        self._st_feature_dict = tfds.features.FeaturesDict(
            {
                'indices': tfds.features.Tensor(shape=(None, None), dtype=tf.int64, encoding='bytes'),
                'values': tfds.features.Tensor(shape=(None, ), dtype=dtype),
                'dense_shape': tfds.features.Tensor(shape=(None, ), dtype=tf.int64),
            },
        )

    def _cast_to_numpy(self, x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        return x

    def _sparse_to_dict(self, example_data):
        return {
            'indices': self._cast_to_numpy(example_data.indices),
            'values': self._cast_to_numpy(example_data.values),
            'dense_shape': self._cast_to_numpy(example_data.dense_shape)
        }

    def _dict_to_sparse(self, example_data_dict):
        return tf.SparseTensor(
            indices = example_data_dict['indices'],
            values = example_data_dict['values'],
            dense_shape = example_data_dict['dense_shape'],
        )

    def get_serialized_info(self):
        return self._st_feature_dict.get_serialized_info()

    def get_tensor_info(self):
        return None

    def encode_example(self, example: tf.SparseTensor):
        example_as_dict = self._sparse_to_dict(example)
        return self._st_feature_dict.encode_example(example_as_dict)

    def decode_example(self, encoded_example: bytes):
        example_as_dict = self._st_feature_dict.decode_example(encoded_example)
        return self._dict_to_sparse(example_as_dict)