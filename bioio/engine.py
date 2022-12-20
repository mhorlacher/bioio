# %%
from multiprocessing.sharedctypes import Value
import yaml
import tensorflow as tf

import bioio

# %%
# Global registry of Loaders and Connectors. 
REGISTERED_CONNECTORS = dict()
REGISTERED_LOADERS = dict()

# %%
class YAMLLoader(yaml.SafeLoader):
    def __init__(self, *args, **kwargs):
        super(YAMLLoader, self).__init__(*args, **kwargs)
        
        # attach prefixed multi-constructor's to the YAML loader. 
        self.add_multi_constructor('!Loader:', self.loader_constructor)
        self.add_multi_constructor('!Connector:', self.connector_constructor)
    
    def loader_constructor(self, loader, tag_suffix, node):
        """Creates a Loader object (i.e. prefixed with '!Loader:'). 
        """
        
        if tag_suffix not in REGISTERED_LOADERS:
            raise ValueError(f'Loader \'{tag_suffix}\' not registered.')
        
        fields = loader.construct_mapping(node, deep=True)
        fields = self._check_and_format_fields(fields, allowed=['args'])
        
        return self._make_loader(tag_suffix, **fields['args'])
    
    def connector_constructor(self, loader, tag_suffix, node):
        """Creates a Connector object (i.e. prefixed with '!Connector:'). 
        """
        
        if tag_suffix not in REGISTERED_CONNECTORS:
            raise ValueError(f'Connector \'{tag_suffix}\' not registered.')
        
        fields = loader.construct_mapping(node, deep=True)
        
        # check that allowed and required fields are met
        fields = self._check_and_format_fields(fields, allowed=['args', 'spec'], required=['spec'])
        
        # Create Connector object from provided 'args' in YAML file
        connector = self._make_connector(tag_suffix, **fields['args'])
        
        # Create TensorSpec object from provided 'spec' in YAML file
        spec = TensorSpec(**fields['spec'])
        
        return ConnectorWrapper(spec, connector)
    
    def _check_and_format_fields(self, fields, allowed=[], required=[]):
        """Ensures that required and allowed fields are met. 

        Args:
            fields (dict): Possibly nested dict. Only the top-level keys are inspected. 
            allowed (list, optional): Keys of allowed fields. Defaults to [].
            required (list, optional): Keys of mandatory fields. Defaults to [].

        Raises:
            ValueError: Raised if non allowed field is encountered. 
            ValueError: Raised if mandatory field is missing. 
        """
        
        # check allowed fields
        for field in fields:
            if field not in allowed: # TODO: Add 'module' field?
                raise ValueError(f'Unexpected field \'{field}\'.')
        
        # check required fields
        for field in required:
            if field not in fields:
                raise ValueError(f'Missing required field \'{field}\'.')
        
        if 'args' not in fields:
            # if no arguments given, add empty args dict
            fields['args'] = {}
        
        return fields
        
    
    def _make_loader(self, tag_suffix, **kwargs):
        """Create Loader instance. 
        """
        return REGISTERED_LOADERS[tag_suffix](**kwargs)
    
    def _make_connector(self, tag_suffix, **kwargs):
        """Create Connector instance. 
        """
        return REGISTERED_CONNECTORS[tag_suffix](**kwargs)
    
    def _make_spec(self, **kwargs):
        """Create TensorSpec instance. 
        """
        return TensorSpec(**kwargs)
    
# %%
class DryYAMLLoader(YAMLLoader):
    """YAML loader for dry-loading the dataspec. 
    
    Loader and Connector constructors are non called and are instead replaced by None objects. 
    """
    
    def _make_loader(self, *args, **kwargs):
        return None
    
    def _make_connector(self, *args, **kwargs):
        return None

# %%
class TensorSpec(tf.TensorSpec):    
    def __init__(self, shape, dtype):
        """Wrapper class around tf.Tensorspec. 

        Args:
            shape (list): Expected shape. 
            dtype (str): Expected dtype. 
        """
        
        super(TensorSpec, self).__init__(shape=tuple(shape), dtype=tf.dtypes.as_dtype(dtype))
    
    def __repr__(self):
        return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype.name})'

# %%
class ConnectorWrapper():
    def __init__(self, spec, connector=None):
        """Wraps a connector object and it's associated TensorSpec. 

        Args:
            spec (TensorSpec): TensorSpec describing the connector's dtype and shape 
            connector (Any, optional): Connector. Defaults to None.
        """
        
        # TensorSpec object
        self.spec = spec
        
        # Connector object
        self.connector = connector
    
    def __repr__(self):
        return f'{type(self).__name__}({type(self.connector).__name__}, {self.spec})'
    
    def __call__(self, *args, **kwargs):
        """Fetches data from connector and asserts it's dtype and shape. 

        Raises:
            ValueError: Raised if no connector is attached (i.e. None). 

        Returns:
            tf.Tensor: Data retrieved from connector. 
        """
        
        if self.connector is None:
            raise ValueError('No connector attached!')
        
        outputs = self.connector(*args, **kwargs)
        outputs = self._cast_dtype(outputs)
        
        # assert outputs shapes (and dtype, although it's already been casted so this is redundant)
        self._assert_spec(outputs)
        
        return outputs

    def _cast_dtype(self, x):
        """Casts the given element to the connector's dtype. 

        Args:
            x (tf.Tensor): Tensor object. 

        Returns:
            tf.Tensor: Tensor object of the connector's dtype. 
        """
        
        return tf.cast(x, dtype=self.spec.dtype)
    
    def _assert_spec(self, x):
        """Asserts that the given element is compatible with the connector's shape and dtype. 

        Args:
            x (tf.Tensor): Tensor object. 
        """
        
        assert self.spec.is_compatible_with(x)

# %%
def connector(cls):
    """Registers a connector class by adding it to bioio.engine.REGISTERED_CONNECTORS. 
    """
    
    REGISTERED_CONNECTORS[cls.__name__] = cls
    return cls

# %%
def load_dataspec(dataspec_yaml, dry=False):
    yaml_loader = YAMLLoader
    if dry:
        yaml_loader = DryYAMLLoader
    
    with open(dataspec_yaml, 'r') as f:
        data = yaml.load(f, yaml_loader)
    
    if dry:
        data_loader = bioio.loaders.DryLoader()
    else:
        data_loader = data['loader']
    
    data_loader.attach_data_structure(data['data_structure'])
    return data_loader

# %%
def load_tfrecords(tfrecords, dataspec_yaml):
    loader = load_dataspec(dataspec_yaml, dry=True)
    return loader.tf_dataset_from_tfrecords(tfrecords)