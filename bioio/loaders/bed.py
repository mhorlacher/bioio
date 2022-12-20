# %%
import pandas as pd
import tensorflow as tf

from bioio.loaders.Base import BaseLoader

# %%
class Bed(BaseLoader):
    def __init__(self, filepath) -> None:
        super(Bed, self).__init__()
        
        self.bed_df = pd.read_csv(filepath, sep='\t', header=None)
        self.bed_df.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand'] + [str(i) for i in range(6, len(self.bed_df.columns))]
        
    def __len__(self):
        return len(self.bed_df)
    
    def __getitem__(self, key_idx):
        return self.fetch_one(**self.bed_df.loc[key_idx].to_dict())

    def __iter__(self):
        for i in range(0, len(self.bed_df)):
            yield self.fetch_one(**self.bed_df.loc[i].to_dict())
        