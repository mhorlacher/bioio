# %%
import argparse
import re
import click

import bioio
import tensorflow as tf

# %%
def make_dict_query_function(keys):
    return eval("lambda x: x" + ''.join([f"[str('{key}')]" for key in keys.split('/')]))

# %%
@click.command()
@click.argument('tfrecords', nargs=-1)
@click.option('-k', '--key-structure', type=str, help='Structure of the key, e.g. \'meta/name\'')
@click.option('-r', '--regex', type=str, help='Regex to filter by')
@click.option('--encoding', type=str, default='bytes')
@click.option('-o', '--output', type=str)
def main(tfrecords, key_structure, regex, encoding, output):
    dict_query_fn = make_dict_query_function(key_structure)

    dataset = bioio.tf.load_tfrecord(list(tfrecords))
    dataset = dataset.filter(lambda x: tf.strings.regex_full_match(dict_query_fn(x), regex))
    bioio.tf.dataset_to_tfrecord(dataset, output, encoding=encoding, pbar=True)

# %%
if __name__ == '__main__':
    main()