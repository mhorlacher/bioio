# %%
from Bio import SeqIO
import tensorflow as tf

# %%
class FastaIterable():
    def __init__(self, filepath, to_upper=True) -> None:
        self.fasta = SeqIO.parse(filepath, 'fasta')
        self.to_upper = to_upper
        
        # for record in SeqIO.parse("example.fasta", "fasta"):
        #     print(record.id)
        
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        for record in self.fasta:
            # header = tf.record.id

            sequence = str(record.seq)
            if self.to_upper:
                sequence = sequence.upper()
            
            yield tf.convert_to_tensor(sequence, dtype=tf.string)