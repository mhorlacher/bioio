# %%
import pandas as pd

# %%
class Bed():
    def __init__(self, filepath) -> None:
        
        self.bed_df = pd.read_csv(filepath, sep='\t', header=None)
        self.bed_df.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand'] + [str(i) for i in range(6, len(self.bed_df.columns))]
        
    def __len__(self):
        return len(self.bed_df)

    def __iter__(self):
        for i in range(0, len(self.bed_df)):
            yield self.bed_df.loc[i].to_dict()