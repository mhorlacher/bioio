# %%
import yaml
import tensorflow as tf
import tensorflow_datasets as tfds

from rbpnet.io import load_typespec

# %%
class TFRecordDataset():
    def __init__(self, tfrecords, dataspec) -> None:
        self.tfrecords = tfrecords
        self.dataspec_file = dataspec
        
        self.nested_typespec = load_typespec(self.dataspec_file)
    
    @property
    def dataset(self):
        pass
