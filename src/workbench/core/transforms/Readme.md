# Workbench Transforms

All classes that tranform an Artifact (a stored entity) to another Artifact are called a **Transforms**. 
There is an `trasform` superclass that specifies a small API used by all subclasses. 

**Transform Abstract API**

```python
@abstractmethod
def input_type(self) -> TransformInput:
    """What Input Type does this Transform Consume"""
    pass

@abstractmethod
def output_type(self) -> TransformOutput:
    """What Output Type does this Transform Produce"""
    pass

@abstractmethod
def set_input(self, resource_url: str):
    """Set the Input for this Transform"""
    pass

@abstractmethod
def set_output_name(self, uuid: str):
    """Set the Output Name (uuid) for this Transform"""
    pass

@abstractmethod
def transform_impl(self):
    """Perform the Transformation from Input to Output"""
    pass
    
@abstractmethod
def get_output(self) -> any:
    """Get the Output from this Transform"""
    pass

@abstractmethod
def validate_input(self) -> bool:
    """Validate the Input for this Transform"""
    pass

@abstractmethod
def validate_output_pre_transform(self) -> bool:
    """Validate, output type, AWS write permissions, etc. before it's created"""
    pass

@abstractmethod
def validate_output(self) -> bool:
    """Validate the Output after it's been created"""
    pass
```

**Stored Entity:** Stored in one or more AWS Services like Data Catalog, Feature Store, Model Registry, etc.