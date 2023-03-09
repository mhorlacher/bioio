# %%
import json

import click
# Add tqdm, once loader.__len__ has been implemented
import tqdm
import tensorflow as tf

from .. import legacy

# %%
def features_to_json_file(features, filepath, indent=2):
    with open(filepath, 'w') as json_file:
        print(json.dumps(features.to_json(), indent=indent), file=json_file)

# %%
@click.command()
@click.argument('dataspec')
@click.option('--gzip', is_flag=True, default=False)
@click.option('--compression-level', type=int, default=None)
@click.option('-t', '--out-tfrecord', required=True, type=str)
@click.option('-f', '--out-features', type=str, default=None)
def main(dataspec, gzip, compression_level, out_tfrecord, out_features):
    loader = legacy.load_dataspec(dataspec)
    loader.summary()
    
    # compress tfrecords to gzip, if flag '--gzip' is set
    tfrecord_options = tf.io.TFRecordOptions(
        compression_type = 'GZIP' if gzip else None,
        compression_level = compression_level if gzip else None,
    )

    # write features spec to json file
    if out_features is None:
        out_features = out_tfrecord.removesuffix('.gz') + '.features.json'
    features_to_json_file(loader._features, out_features)
    
    with tf.io.TFRecordWriter(out_tfrecord, tfrecord_options) as tfrecord, tqdm.tqdm(total=len(loader)) as pbar:
        for example in iter(loader):
            tfrecord.write(loader.serialize(example))
            pbar.update(1)

# %%
if __name__ == '__main__':
    main()