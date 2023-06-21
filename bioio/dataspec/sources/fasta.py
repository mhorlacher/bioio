# %%
from Bio import SeqIO
import tensorflow as tf

# %%
class FastaIterable():
    def __init__(self, filepath, return_header=True, to_upper=True) -> None:
        self.filepath = filepath
        self.return_header = return_header
        self.to_upper = to_upper
        
        # for record in SeqIO.parse("example.fasta", "fasta"):
        #     print(record.id)
        
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        for record in SeqIO.parse(self.filepath, 'fasta'):
            return_record = {}

            if self.return_header:
                # header_id = tf.convert_to_tensor(record.id, dtype=tf.string)
                # header_desc = tf.convert_to_tensor(record.description, dtype=tf.string)
                # return_record['header_id'] = header_id
                # return_record['header_desc'] = header_desc
                return_record['header'] = tf.convert_to_tensor(record.description, dtype=tf.string)

            sequence = str(record.seq)
            if self.to_upper:
                sequence = sequence.upper()
                return_record['sequence'] = tf.convert_to_tensor(sequence, dtype=tf.string)
            
            yield return_record