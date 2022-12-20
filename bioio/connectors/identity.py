# %%
from bioio.engine import connector

# %%
@connector
class IdentityBedName:
    def __init__(self, format_string='{chrom}:{start}-{end}:{strand}') -> None:
        self.format_string = format_string
    
    def __call__(self, **kwargs) -> str:
        return self.format_string.format(**kwargs)