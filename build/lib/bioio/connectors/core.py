# %%
import yaml
import tensorflow as tf

# %%
def register_connector(tag=None):
    def func(cls):
        def constructor(loader, node):
            # get fields (and subfield, i.e. entire sub-tree)
            fields = loader.construct_mapping(node, deep=True)
            
            # Create new Connector (cls) and TensorSpec, then return DataConnector
            if ConnectorWrapper._load_connector:
                return ConnectorWrapper(TensorSpec(**fields['spec']), cls(**fields['args']))
            else:
                # Do NOT load the connector - just the TypeSpec. This is useful when 
                # one wants to only load the dataspec to get insight into the dataspec 
                # data structure, but does not want to load the Connector (e.g. the filepath 
                # does not exist anymore, etc.). 
                # TODO: Find a less hacky solution. 
                return ConnectorWrapper(TensorSpec(**fields['spec']))
            
        
        # set tag (need to set non-local, because we are assigning to it and thus Python 
        # will look for it in the local scope - where it is not to be found). 
        nonlocal tag
        if tag is None:
                tag = cls.__name__
        else:
            tag = tag.strip('!')
        tag = '!' + tag
        
        yaml.add_constructor(tag, constructor)
        #print('Registered:', tag)
        return cls
    return func

# %%
class TensorSpec(tf.TensorSpec):
    def __init__(self, shape, dtype):
        super(TensorSpec, self).__init__(shape=tuple(shape), dtype=tf.dtypes.as_dtype(dtype))

# %%
class ConnectorWrapper():
    _load_connector = True
    
    def __init__(self, spec, connector=None):
        self.spec = spec
        self.connector = connector
    
    def __repr__(self):
        return f'{type(self).__name__}({type(self.connector).__name__}, shape={self.spec.shape}, dtype={self.spec.dtype.name})'
    
    def __call__(self, *args, **kwargs):
        #print(args, kwargs)
        
        if self.connector is None:
            raise ValueError('No connector attached!')
        
        outputs = self.connector(*args, **kwargs)
        outputs = self._cast_dtype(outputs)
        
        # assert outputs shapes (and dtype, although it's already been casted so this is redundant)
        self._assert_spec(outputs)
        
        return outputs

    def _cast_dtype(self, x):
        return tf.cast(x, dtype=self.spec.dtype)
    
    def _assert_spec(self, x):
        assert self.spec.is_compatible_with(x)