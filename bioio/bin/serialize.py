
# %%
import os

import click
import tqdm
import tensorflow as tf

from bioio import load_biospec
from bioio.tf.utils import dataset_to_tensor_features, features_to_json_file, serialize_dataset

# %%
@click.command()
@click.argument('biospec')
@click.option('--gzip', is_flag=True, default=False)
@click.option('--gzip-compression-level', type=int, default=None)
@click.option('--encoding', type=str, default='bytes')
@click.option('-d', '--directory', type=str, default=None)
@click.option('-t', '--out-tfrecord', required=True, type=str)
@click.option('-f', '--out-features', type=str, default=None)
@click.option('--just-write-features', is_flag=True, default=False, help='Just write features and exit.')
def main(biospec, gzip, gzip_compression_level, encoding, out_tfrecord, out_features, directory, just_write_features):
    if directory is not None:
        # change working directory (all relative paths in biospec.yml will be relative to this directory)
        if not os.path.isdir(directory):
            raise ValueError(f"'{directory}' is not a directory.")
        os.chdir(directory)

    dataset = load_biospec(biospec)
    print(dataset.element_spec)

    features = dataset_to_tensor_features(dataset, encoding=encoding)

    # write features spec to json file
    if out_features is None:
        out_features = out_tfrecord + '.features.json'
    features_to_json_file(features, out_features)

    if just_write_features:
        # just write features and exit
        return
    
    # compress tfrecords to gzip, if flag '--gzip' is set
    tfrecord_options = tf.io.TFRecordOptions(
        compression_type = 'GZIP' if gzip else None,
        compression_level = gzip_compression_level if gzip else None,
    )
    
    # determine dataset cardinality 
    cardinality = dataset.cardinality()
    cardinality = int(cardinality) if cardinality > 0 else None

    with tf.io.TFRecordWriter(out_tfrecord, tfrecord_options) as tf_writer, tqdm.tqdm(total=cardinality) as pbar:
        # use features to serialize examples to binary string
        serialized_dataset = serialize_dataset(dataset, features)

        # write serialized examples to tfrecord
        for serialized_example in serialized_dataset:
            tf_writer.write(serialized_example)
            pbar.update(1)

# %%
if __name__ == '__main__':
    main()