# %%
import tensorflow as tf

from bioio.tf.utils import multi_hot, better_py_function_kwargs

# %%
class BedFormatName:
    tensor_spec = tf.TensorSpec(shape=(), dtype=tf.string)

    def __init__(self, format_string='{chrom}:{start}-{end}:{strand}') -> None:
        self.format_string = format_string
    
    def format(self, **kwargs):
        return self.format_string.format(**kwargs)

    def __call__(self, example) -> str:
        tensor = better_py_function_kwargs(Tout=self.tensor_spec)(self.format)(example)
        # tensor = self.format_string.format(**kwargs)
        tensor.set_shape(self.tensor_spec.shape)
        return tensor

# %%
class BedColumnSparseLabels:
    def __init__(self, column, sep=','):
        self._column = column
        self._sep = sep

    def __call__(self, example):
        label_string = example[self._column]
        label_string.set_shape(())
        return tf.cast(tf.strings.to_number(tf.strings.split(label_string, sep=self._sep)), tf.int32)

# %%
class BedColumnMultihotLabels:
    tensor_spec = tf.TensorSpec(shape=(None, ), dtype=tf.int64)

    def __init__(self, column, depth, sep=','):
        self._column = column
        self._sep = sep
        self._depth = depth

    def __call__(self, example):
        label_string = example[self._column]
        label_string.set_shape(())
        label_indices = tf.cast(tf.strings.to_number(tf.strings.split(label_string, sep=self._sep)), tf.int32)
        return multi_hot(label_indices, self._depth)