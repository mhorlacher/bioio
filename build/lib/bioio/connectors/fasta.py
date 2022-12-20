# %%
import pysam

from bioio.engine import connector
from bioio.utils import reverse_complement, sequence2onehot

# %%
@connector
class Fasta():
    def __init__(self, filepath) -> None:
        self._fasta = pysam.FastaFile(filepath)
    
    def fetch(self, chrom, start, end, strand='+', **kwargs):
        sequence = self._fasta.fetch(chrom, start, end)

        if strand == '+':
            pass
        elif strand == '-':
            sequence = ''.join(reverse_complement(sequence))
        else:
            raise ValueError(f'Unknown strand: {strand}')
        
        return sequence2onehot(sequence)

    def __call__(self, *args, **kwargs):
        return self.fetch(*args, **kwargs)

