# %%
import json

import tensorflow as tf
import tensorflow_datasets as tfds

# %%
# def get_structure_types(structure):
#     structure_tensor = tf.nest.map_structure(tf.constant, structure)
#     return tf.nest.map_structure(lambda x: x.dtype, structure_tensor)

def get_structure_signature(structure):
    structure_tensor = tf.nest.map_structure(tf.constant, structure)
    return tf.nest.map_structure(lambda x: tf.TensorSpec(shape=x.shape, dtype=x.dtype, name=None), structure_tensor)

# %%
def get_generalized_structure_signature(structure):
    structure_tensor = tf.nest.map_structure(tf.constant, structure)
    return tf.nest.map_structure(lambda x: tf.TensorSpec(shape=[None]*len(x.shape), dtype=x.dtype, name=None), structure_tensor)

# %%
def dataset_from_iterable(py_iterable):
    output_signature = get_structure_signature(next(iter(py_iterable)))
    return tf.data.Dataset.from_generator(lambda: iter(py_iterable), output_signature=output_signature)

# %%
def tensorspec_to_tensor_feature(tensorspec, encoding=None, **kwargs):
    if tensorspec.dtype is tf.string or encoding is None:
        encoding = tfds.features.Encoding.NONE

    return tfds.features.Tensor(shape=tensorspec.shape, dtype=tensorspec.dtype, encoding=encoding, **kwargs)

# %%
# def dataset_to_tensor_features(dataset, **kwargs):
#     return tfds.features.FeaturesDict(
#         tf.nest.map_structure(
#             lambda spec: tensorspec_to_tensor_feature(spec, **kwargs), 
#             dataset.element_spec))

# %%
def dataset_to_tensor_features(dataset, **kwargs):
    first_example = next(iter(dataset))
    # first_example_signature = get_structure_signature(first_example)
    first_example_signature = get_generalized_structure_signature(first_example)
    return tfds.features.FeaturesDict(
        tf.nest.map_structure(
            lambda spec: tensorspec_to_tensor_feature(spec, **kwargs), 
            first_example_signature))

# %%
def features_to_json_file(features, filepath, indent=2):
    with open(filepath, 'w') as json_file:
        print(json.dumps(features.to_json(), indent=indent), file=json_file)

# %%
def features_from_json_file(filepath):
    with open(filepath) as json_file:
        features = tfds.features.FeaturesDict.from_json(json.load(json_file))
    return features

# %%
def serialize_dataset(dataset, features):
    for example in dataset:
        yield features.serialize_example(tf.nest.map_structure(lambda e: e.numpy(), example))

# %%
def dataset_to_tfrecord(dataset, filepath, encoding='bytes'):
    features = dataset_to_tensor_features(dataset, encoding=encoding)
    features_to_json_file(features, filepath + '.features.json')

    with tf.io.TFRecordWriter(filepath) as tfrecord_write:
        for serialized_example in serialize_dataset(dataset, features):
            tfrecord_write.write(serialized_example)

# %%
def deserialize_dataset(dataset, features):
    return dataset.map(features.deserialize_example)

# %%
def load_tfrecord(tfrecord_file, features_file=None, deserialize=True):
    dataset = tf.data.TFRecordDataset([tfrecord_file])

    if deserialize:
        if features_file is None:
            features_file = tfrecord_file + '.features.json'
        features = features_from_json_file(features_file)
        dataset = deserialize_dataset(dataset, features)

    return dataset

# %%
def tensor_to_numpy(tensor):
    return tensor.numpy()

# %%
def decode_if_bytes(x):
    if isinstance(x, bytes):
        x = x.decode('UTF-8')
    return x

# %%
def tensor_to_numpy_and_decode_if_bytes(x):
    x = tf.nest.map_structure(tensor_to_numpy, x)
    x = tf.nest.map_structure(decode_if_bytes, x)
    return x

# %%
def dtype_to_tensor_spec(v):
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v

# %%
def tensor_spec_to_dtype(v):
  return v.dtype if isinstance(v, tf.TensorSpec) else v

# %%
def py_function_nest(func, inp, Tout, name=None):
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(inp, flat_inp, expand_composites=True)
        out = func(*reconstructed_inp)
        return tf.nest.flatten(out, expand_composites=True)

    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func, 
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[tensor_spec_to_dtype(v) for v in flat_Tout],
        name=name)
    spec_out = tf.nest.map_structure(dtype_to_tensor_spec, Tout, expand_composites=True)
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out

# %%
def better_py_function_kwargs(Tout, numpy=True, decode_bytes=True):
    def decorator(func):
        def func_wrapper(kwargs):
            if numpy:
                kwargs = tf.nest.map_structure(tensor_to_numpy, kwargs)
            if decode_bytes:
                kwargs = tf.nest.map_structure(decode_if_bytes, kwargs)
            return func(**kwargs)
        return lambda kwargs: py_function_nest(func_wrapper, inp=[kwargs], Tout=Tout)
    return decorator

# %%
# def pyfunc(output_types, numpy=True, decode_bytes=True, expand_kwargs=True):
#     def decorator(func):
#         def func_wrapper(inp):
#             if numpy:
#                 inp = tf.nest.map_structure(tensor_to_numpy, inp)
#             if decode_bytes:
#                 inp = tf.nest.map_structure(decode_if_bytes, inp)
            
#             if expand_kwargs:
#                 return func(**inp)
#             else:
#                 return func(inp)
#         return lambda x: py_function_nest(func_wrapper, inp=[x], Tout=output_types)
#     return decorator

# %%
def multi_hot(x, depth):
    return tf.reduce_sum(tf.one_hot(x, depth=depth, dtype=tf.int64), axis=0)