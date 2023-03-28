# %%
import struct

import click

# %%
def count_records(tfrecord):
    count = 0
    with open(tfrecord, 'rb') as tfr:
        while True:
            try:
                # byte length
                byte_len = tfr.read(8)
                if len(byte_len) == 0:
                    break

                # crc length
                proto_len = struct.unpack('q', byte_len)[0]

                # crc length + proto + crc
                _ = tfr.read(4 + proto_len + 4)

                count +=1
            except Exception:
                print('Not a valid TFRecord file.')
                break
        
    return count

# %%
@click.command()
@click.argument('tfrecord')
def main(tfrecord):
    print(count_records(tfrecord))

# %%
if __name__ == '__main__':
    main()