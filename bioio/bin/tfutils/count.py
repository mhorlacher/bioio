# %%
import click
import tensorflow as tf

# %%
@click.command()
@click.argument('tfrecord')
def main(tfrecord):
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord):
        count += 1
    print(count)

# %%
if __name__ == '__main__':
    main()