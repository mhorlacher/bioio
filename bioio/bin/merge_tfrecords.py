# %%
import click
import tqdm
import tensorflow as tf

from ..dataspec import load_tfrecord
from ..dataspec.dataset_ops import dataset_to_tensor_features, features_to_json_file, serialize_dataset

# %%
def merge_dicts(dicts):
    xm = dicts[0]
    for x in dicts[1:]:
        xm = {**xm, **x}
    return xm

# %%
@click.command()
@click.argument('tfrecords', nargs=-1)
@click.option('--gzip', is_flag=True, default=False)
@click.option('--compression-level', type=int, default=None)
@click.option('-t', '--out-tfrecord', required=True, type=str)
@click.option('-f', '--out-features', type=str, default=None)
def main(tfrecords, gzip, compression_level, out_tfrecord, out_features):
    # load datasets
    datasets = tuple([load_tfrecord(tfrecord) for tfrecord in tfrecords])

    dataset = tf.data.Dataset.zip(datasets)
    dataset = dataset.map(lambda *dicts: merge_dicts(dicts))
    print(dataset.element_spec)
    
    # compress tfrecords to gzip, if flag '--gzip' is set
    tfrecord_options = tf.io.TFRecordOptions(
        compression_type = 'GZIP' if gzip else None,
        compression_level = compression_level if gzip else None,
    )
    
    # determine dataset cardinality 
    cardinality = dataset.cardinality()
    cardinality = int(cardinality) if cardinality > 0 else None

    with tf.io.TFRecordWriter(out_tfrecord, tfrecord_options) as tf_writer, tqdm.tqdm(total=cardinality) as pbar:
        features = dataset_to_tensor_features(dataset, encoding='zlib')

        # write features spec to json file
        if out_features is None:
            out_features = out_tfrecord.removesuffix('.gz') + '.features.json'
        features_to_json_file(features, out_features)

        # use features to serialize examples to binary string
        serialized_dataset = serialize_dataset(dataset, features)

        # write serialized examples to tfrecord
        for serialized_example in serialized_dataset:
            tf_writer.write(serialized_example)
            pbar.update(1)

# %%
if __name__ == '__main__':
    main()