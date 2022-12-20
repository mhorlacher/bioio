# %%
import sys

import numpy as np
import pyBigWig

from bioio.engine import connector
from bioio.utils import nan_to_zero

# %%
@connector
class BigWig():
    def __init__(self, bigwig_filepath) -> None:
        self._bigWig = pyBigWig.open(bigwig_filepath)
    
    def values(self, chrom, start, end, **kwargs):
        # # lazy-load bigWig
        # if not isinstance(self._bigWig, pyBigWig.pyBigWig):
        #     self._bigWig = pyBigWig.open(self._bigWig)
        
        try:
            values = self._bigWig.values(chrom, start, end)
        except RuntimeError as e:
            # TODO: Use logging module instead. This allows users to turn off warnings. 
            print('WARNING: ' + str(e), file=sys.stderr)
            print(f'WARNING: ({chrom}\t{start}\t{end})', file=sys.stderr)
            return [0.0]*(end-start)
        
        values = [nan_to_zero(v) for v in values]        
        return values

    def __call__(self, *args, **kwargs):
        return self.values(*args, *kwargs)

# %%
@connector
class StrandedBigWig():
    def __init__(self, bigwig_plus, bigwig_minus, reverse_minus=True) -> None:
        self._bigWig_plus = BigWig(bigwig_plus)
        self._bigWig_minus = BigWig(bigwig_minus)
        self.reverse_minus = reverse_minus

    def values(self, chrom, start, end, strand='+', **kwargs):
        """Returns values for a given range and strand. 
        
        Args:
            chrom  (str): Chromosome (chr1, chr2, ...)
            start  (int): 0-based start position
            end    (int): 0-based end position
            strand (str): Strand ('+' or '-')
            
        Returns:
            numpy.array: Numpy array of shape (end-start, )
        """

        if strand == '+':
            bigWig = self._bigWig_plus
        elif strand == '-':
            bigWig = self._bigWig_minus
        else:
            raise ValueError(f'Unexpected strand: {strand}')

        values = bigWig.values(chrom, start, end)

        if strand == '-' and self.reverse_minus:
            values = list(reversed(values))

        return np.array(values, dtype=np.float32)

    def __call__(self, *args, **kwargs):
        return self.values(*args, **kwargs)
        