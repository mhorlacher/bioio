# %%
import sys
import types
import importlib
import pathlib

import yaml
import tensorflow as tf

# %%
from bioio.tf.utils import dataset_from_iterable

# %%
# To import modules from the current directory
# TODO: Give user option to add arbitrary paths to sys.path
sys.path.append('.')

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
        self.add_constructor('!PyIterable', self.constructor_iterable)
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

        transform_object = import_object_from_string(fields['object'])
        if isinstance(transform_object, types.FunctionType):
            transform_name = transform_object.__name__
            # if transform is a function, we need to inject arguments
            transform = lambda x: transform_object(x, **fields['args'])
            transform.__name__ = transform_name
        else:
            # if transform is a class, we need to instantiate it
            transform = transform_object(**fields['args'])

            if not hasattr(transform, '__name__'):
                transform_name = type(transform).__name__
            else:
                transform_name = transform.__name__
            transform.__name__ = transform_name

            # assert that the transform is callable (it could be an abritrary object)
            assert callable(transform), f'Transform {fields["object"]} is not callable'

        # # optionally, cast the output of the transform to a specific tensorflow dtype
        if 'dtype' in fields:
            try:
                dtype = tf.dtypes.as_dtype(fields['dtype'])
            except TypeError:
                raise TypeError(f"Invalid dtype: {fields['dtype']}")
            # if dtype is specified, we need to cast the output of the transform
            transform_ = transform # this is to avoid infinite recursion
            transform = lambda x: tf.cast(transform_(x), dtype)
            transform.__name__ = transform_.__name__

        if 'input' not in fields:
            # if input is not specified, we assume the transform is called within another 
            # transform and return the transform itself
            return transform

        # if input is not a tf.data.Dataset, we assume it's a nested structure of 
        # tf.data.Dataset and zip them together to create a zipped tf.data.Dataset
        if not isinstance(fields['input'], tf.data.Dataset):
            fields['input'] = tf.data.Dataset.zip(fields['input'], name='zip')

        # map the transform over the input dataset
        dataset = fields['input'].map(transform, name=transform.__name__)

        # optionally, apply an additional stack of zero or more transforms to the dataset
        if 'map' in fields:
            for transform in fields['map']:
                if isinstance(transform, tf.data.Dataset):
                    raise TypeError('Expected a transform, but got a tf.data.Dataset. Did specify \'input\' in a nested context?')
                dataset = dataset.map(transform, name=transform.__name__)

        # TODO: Replace with 
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# %%
def load_biospec(biospec_yaml, to_dataset=True):
    with open(biospec_yaml, 'r') as f:
        data = yaml.load(f, BiospecLoader)['data']
    
    if to_dataset:
        data = tf.data.Dataset.zip(data)

    return data