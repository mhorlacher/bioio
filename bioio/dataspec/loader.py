# %%
import importlib

import yaml
import tensorflow as tf

# %%
from bioio.tf.ops import dataset_from_iterable

# %%
def import_object_from_string(import_string):
    import_string = import_string.strip()
    module_name, object_name = '.'.join(import_string.split('.')[:-1]), import_string.split('.')[-1]
    module = importlib.import_module(module_name, object_name)
    return module.__getattribute__(object_name)

# %%
class BiospecLoader(yaml.SafeLoader):
    def __init__(self, *args, **kwargs):
        super(BiospecLoader, self).__init__(*args, **kwargs)
        
        # add constructors to Loader and associate with tag
        self.add_constructor('!Iterable', self.constructor_iterable)
        self.add_constructor('!Transform', self.constructor_transform)
    
    # def constructor_source(self, loader, node):
    #     fields = fields = loader.construct_mapping(node, deep=True)
        
    #     if 'args' not in fields:
    #         fields['args'] = {}

    #     source = import_object_from_string(fields['source'])(**fields['args'])

    #     dataset = source()
    #     return dataset

    def constructor_iterable(self, loader, node):
        fields = fields = loader.construct_mapping(node, deep=True)
        
        if 'args' not in fields:
            fields['args'] = {}

        iterable = import_object_from_string(fields['object'])(**fields['args'])

        return dataset_from_iterable(iterable)

    def constructor_transform(self, loader, node):
        fields = fields = loader.construct_mapping(node, deep=True)

        if 'args' not in fields:
            fields['args'] = {}

        transform = import_object_from_string(fields['object'])

        if not isinstance(fields['input'], tf.data.Dataset):
            fields['input'] = tf.data.Dataset.zip(fields['input'], name='zip')
        
        dataset = fields['input'].map(transform(**fields['args']), name=transform.__name__)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# %%
def load_biospec(biospec_yaml, to_dataset=True):
    with open(biospec_yaml, 'r') as f:
        data = yaml.load(f, BiospecLoader)['data']
    
    if to_dataset:
        data = tf.data.Dataset.zip(data)

    return data