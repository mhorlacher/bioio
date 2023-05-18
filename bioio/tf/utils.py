# %%
import sys
import json

import tqdm
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
    structure_tensor = tf.nest.map_structure(tf.convert_to_tensor, structure)
    # print(structure_tensor)
    return tf.nest.map_structure(lambda x: tf.TensorSpec(shape=[None]*len(x.shape), dtype=x.dtype, name=None), structure_tensor)

# %%
def dataset_from_iterable(py_iterable):
    # output_signature = get_structure_signature(next(iter(py_iterable)))
    output_signature = get_generalized_structure_signature(next(iter(py_iterable)))
    # print(output_signature)
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
def load_tfrecord(tfrecords, features_file=None, deserialize=True, shuffle=None):
    if isinstance(tfrecords, str):
        # backward compatibility, accept a single tfrecord file instead of a list of tfrecord files
        tfrecords = [tfrecords]
    dataset = tf.data.Dataset.from_tensor_slices(tfrecords)
    dataset = dataset.interleave(lambda fp: tf.data.TFRecordDataset(fp), cycle_length=1, block_length=1, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle is not None:
        # shuffle examples before deserializing
        assert isinstance(shuffle, int)
        dataset = dataset.shuffle(shuffle)

    # optimize IO
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # desrialize examples using feature specification in auxiliary file
    if deserialize:
        if features_file is None:
            # if list of tfrecords is supplied but no features file, use features file of first tfrecord - this must exist
            features_file = tfrecords[0] + '.features.json'
        features = features_from_json_file(features_file)

        dataset = dataset.map(features.deserialize_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # optimize IO
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

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

# %%
def estimate_record_size(tfrecords, take=None):
    """
    Estimate mean and standard deviation of TFRecord record sizes in bytes.
    """

    dataset = load_tfrecord(tfrecords, deserialize=False)
    if take is not None:
        dataset = dataset.take(take)

    sizes = []
    print('Estimating record size...', file=sys.stderr)
    for record in tqdm.tqdm(dataset.as_numpy_iterator()):
        sizes.append(tf.cast(sys.getsizeof(record), dtype=tf.float32))
    return tf.math.reduce_mean(sizes), tf.math.reduce_std(sizes)

# %%
def get_max_buffer_size_for_target_memory(record_size_mean, record_size_std, memory_in_mb=1024):
    """
    Estimate the maximum viable buffer sizes for a given memory budget. 
    """
    return int(1000*1000*memory_in_mb / (record_size_mean + 2*record_size_std))

# %%
def get_max_buffer_size(tfrecords, memory_in_mb=1024, take=None):
    """
    Estimate the maximum viable buffer sizes for a given memory budget. 
    """
    record_size_mean, record_size_std = estimate_record_size(tfrecords, take=take)
    return get_max_buffer_size_for_target_memory(record_size_mean, record_size_std, memory_in_mb)