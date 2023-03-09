# %%
import click

from ..dataspec.index import index_tfrecord
from ..dataspec.dataset_ops import features_from_json_file

def deserialize_and_get_nested_values(proto, keys, features):
    x = features.deserialize_example(proto)
    for key in keys.split('/'):
        x = x[key]
    return x.numpy().decode('UTF-8')

# %%
@click.command()
@click.argument('tfrecord')
@click.option('-i', '--index', default=None)
@click.option('-k', '--key', default=None)
def main(tfrecord, index, key):
    if index is None:
        index = tfrecord + '.idx'
    
    proto_fn = None
    if key is not None:
        features = features_from_json_file(tfrecord + '.features.json')
        proto_fn = lambda proto: deserialize_and_get_nested_values(proto, key, features)
    
    index_tfrecord(tfrecord, index, pbar=True, proto_fn=proto_fn)

# %%
if __name__ == '__main__':
    main()