# %%
import click

# %%
from . import __version__
from .bin import serialize, biospec2dot, tfrecord2idx, merge_tfrecords, tfutils

# %%
@click.group()
@click.version_option(__version__)
def main():
    pass

# %%
main.add_command(serialize.main, name='serialize')
main.add_command(biospec2dot.main, name='biospec2dot')
main.add_command(tfrecord2idx.main, name='tfrecord2idx')
main.add_command(merge_tfrecords.main, name='merge-tfrecords')
main.add_command(tfutils.main, name='tf-utils')

# %%
if __name__ == '__main__':
    main()
