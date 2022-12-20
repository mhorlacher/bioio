# %%
import click
# Add tqdm, once loader.__len__ has been implemented
import tqdm
import tensorflow as tf

import bioio

# %%
@click.command()
@click.argument('dataspec')
@click.option('--compression', type=str, default=None)
@click.option('--compression-level', type=int, default=None)
@click.option('-o', '--output', required=True, type=str)
def main(dataspec, compression, compression_level, output):
    loader = bioio.load_dataspec(dataspec)
    loader.summary()
    
    tfrecord_options = tf.io.TFRecordOptions(
        compression_type = compression.upper() if compression else None,
        compression_level = compression_level.upper() if compression else None,
    )
    
    with tf.io.TFRecordWriter(output, tfrecord_options) as tfrecord, tqdm.tqdm(total=len(loader)) as pbar:
        for example in iter(loader):
            tfrecord.write(loader.serialize(example))
            pbar.update(1)

# %%
if __name__ == '__main__':
    main()