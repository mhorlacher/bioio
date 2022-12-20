# %%
import sys
import json

import tensorflow as tf
import tensorflow_datasets as tfds

from bioio.engine import REGISTERED_LOADERS

# %%
class MetaLoader(type):
    """Meta-class for registration of Loaders with yaml. 
    """
    
    def __new__(cls, name, bases, dct):
        # create a new Loader class
        new_cls = super().__new__(cls, name, bases, dct)
        # register the loader to the global registry
        REGISTERED_LOADERS[new_cls.__name__] = new_cls
        return new_cls

# %%
class BaseLoader(metaclass=MetaLoader):
    """Base loader - all loaders inherit from this class. 
    """
    
    def __init__(self) -> None:
        # nested structure of Connector objects
        self._data_structure = None
    
    def __call__(self, *args, **kwargs): 
        # Code now moved to 'self.fetch_one'. Hopefully this makes things more explicit. 
        raise NotImplementedError()
    
    def _fetch_function(self, connector, *args, **kwargs):
        """Fetches an element from a given connector. 
        
        Overwriting this function can customize how elements are fetched from connectors. For instance,
        it allows to provide differnet inputs to calls of different connectors or element post-processing. 

        Args:
            connector (io.connectors.ConnectorWrapper): Connector. 

        Returns:
            Any: Element fetched from Connector. 
        """
        
        return connector(*args, **kwargs)

    def fetch_one(self, *args, **kwargs):
        """Returns a sample instance of the (nested) data structure. 
        
        The single samples are pulled from all atomic Connector objects of the data 
        structure by passing *args and **kwargs to the Connector object's call method. 
        It even allows to add the loader's arguments to the samples (e.g. in an INFO field) 
        by post-processing the sample dict. 

        Returns:
            dict: Sample instance. 
        """
        
        return tf.nest.map_structure(lambda connector: self._fetch_function(connector=connector, *args, **kwargs), self.data_structure)
        
    
    def __getitem__(self, key_idx):
        """User-implemented function for random sample access. 

        Args:
            key_idx (int or str): Key or index. 

        Raises:
            NotImplementedError: Only implemented by subclasses. 
        """
        
        raise NotImplementedError()
    
    def __iter__(self):
        """User-implemented function for iterative sample-access. 

        Raises:
            NotImplementedError: Only implemented by subclasses. 
        """
        
        raise NotImplementedError()
    
    def __len__(self):
        return None
    
    def serialize(self, x):
        """Serializes an example dict to bytes. 

        Serialization is achieved by first serializing each atom of the nested structure 
        via tf.io.serialize_tensor and then serializing the resulting structure via 
        tensorflow_datasets FeatureDict. 

        Args:
            x (dict): Example dict. 

        Returns:
            str: Byte-string. 
        """
        
        x = tf.nest.map_structure(lambda y: tf.io.serialize_tensor(y).numpy(), x)
        return self._features_serialized.serialize_example(x)
    
    def deserialize(self, x):
        x = self._features_serialized.deserialize_example(x)
        return tf.nest.map_structure(tf.io.parse_tensor, x, self.dtypespec)
    
    @property
    def tf_dataset(self):
        """Returns an instance of tf.data.Dataset for the Loader's data structure. 
        """
        
        # define tf.data.Dataset output signature (i.e. a nested structure of tf.TypeSpec objects
        # obtained from each connector). 
        output_signature = tf.nest.map_structure(lambda connector: connector.spec, self.data_structure)
        return tf.data.Dataset.from_generator(self.__iter__, output_signature=output_signature)

    def tf_dataset_from_tfrecords(self, tfrecords, **kwargs):
        """Load a list of TFRecord files and deserialize them with the Loader's data structure. 

        Args:
            tfrecords (list): List of TFRecord filepaths. 

        Returns:
            tf.data.Dataset: Dataset which yields the deserialized nested data structure. 
        """
        
        # load serialized examples from TFRecord files
        dataset = tf.data.TFRecordDataset(tfrecords, **kwargs)
        # deserialize examples
        dataset = dataset.map(self.deserialize)
        return dataset

    def summary(self, file_handle=None):
        """Prints the Loader's data structure. 
        """
        
        if file_handle is None:
            file_handle = sys.stdout
        
        # pretty-print the nested dict. 
        print(json.dumps(tf.nest.map_structure(str, self.data_structure), indent=2), file=file_handle)
    
    @property
    def typespec(self):
        """Returns the Loader's nested tf.Typespec(s). 
        """
        
        return tf.nest.map_structure(lambda x: x.spec, self.data_structure)
    
    @property
    def dtypespec(self):
        """Returns the Loader's nested tf.dtypes.DType(s). 
        """
        
        return tf.nest.map_structure(lambda x: x.spec.dtype, self.data_structure)
    
    @property
    def data_structure(self):
        if self._data_structure is None:
            raise RuntimeError("No data structure loaded. Use 'load()' before calling data-dependent functions.")
        return self._data_structure
    
    def attach_data_structure(self, data_structure):
        self._data_structure = data_structure
        self._features = self._make_features(self.data_structure)
        self._features_serialized = self._make_features_dtype(self.data_structure, tf.string)
    
    def _make_features(self, data_structure):
        """Returns a FeatureDict for the given data_structure. 

        Args:
            data_structure (dict): Data structure. 

        Returns:
            tfds.features.FeaturesDict: FeatureDict. 
        """
        
        return tfds.features.FeaturesDict(tf.nest.map_structure(lambda loader: loader.spec.dtype, data_structure))
    
    def _make_features_dtype(self, data_structure, dtype):
        """Return a FeatureDict, with all atoms set to the given dtype. 

        Args:
            structurdata_structuree (dict): Data structure. 
            dtype (tf.dtypes.DType): Data type. 

        Returns:
            tfds.features.FeaturesDict: FeatureDict. 
        """
        
        return tfds.features.FeaturesDict(tf.nest.map_structure(lambda _: dtype, data_structure))

# %%
class DryLoader(BaseLoader):
    pass