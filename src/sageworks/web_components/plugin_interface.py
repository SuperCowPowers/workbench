"""An abstract class that defines the web component interface for SageWorks"""
from abc import abstractmethod
from typing import Any, Union, get_origin, get_args
from enum import Enum
import pandas as pd
from dash import dcc
import plotly.graph_objects as go

# Local Imports
from sageworks.web_components.component_interface import ComponentInterface


class PluginType(Enum):
    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"


class PluginInputType(Enum):
    DATA_SOURCE_DETAILS = "data_source_details"
    FEATURE_SET_DETAILS = "feature_set_details"
    MODEL_DETAILS = "model_details"
    ENDPOINT_DETAILS = "endpoint_details"


class PluginInterface(ComponentInterface):
    """A Web Plugin Interface
    Notes:
      - These methods are ^stateless^, all data should be passed through the
        arguments and the implementations should not reference 'self' variables
      - The 'create_component' method must be implemented by the child class
      - The 'generate_component_figure' is optional (some components don't use Plotly figures)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "plugin_type") or not isinstance(cls.plugin_type, PluginType):
            raise TypeError("Subclasses must define a 'plugin_type' of type PluginType")

        if not hasattr(cls, "plugin_input_type") or not isinstance(cls.plugin_input_type, PluginInputType):
            raise TypeError("Subclasses must define a 'plugin_input_type' of type PluginInputType")
        
    
    # If any base class method or parameter is missing from a subclass, or if a subclass method parameter is not correctly typed,
    # a call of issubclass(subclass, cls) will return False, allowing runtime checks for plugins
    # An 'assert issubclass(subclass, cls) call could be implemented in the plugin loader interface or the test plugin_interface_test.py
    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is PluginInterface:
            subclass_dict = subclass.__mro__[0].__dict__
            cls_dict = cls.__mro__[0].__dict__
            cls_abstract_methods = cls.__abstractmethods__

            # Iterate through base class methods
            for method_name, method in cls_dict.items():

                # If base class method is abstract method
                if method_name in cls_abstract_methods and hasattr(method, '__annotations__'):

                    # If base class method not in subclass
                    if method_name not in subclass_dict:
                        # print(f"Method {method_name} not present in subclass.")
                        return False
                    
                    # For each parameter in subclass method dictionary
                    for parameter in subclass_dict[method_name].__annotations__:
                        
                        sub_param_type = subclass_dict[method_name].__annotations__[parameter]
                        # print(f"Subclass parameter {parameter} of method {method_name} is of type {sub_param_type}")

                        # If subclass method param not in base class method params
                        if parameter not in cls_dict[method_name].__annotations__:
                            # print(f"Subclass parameter {parameter} of method {method_name} not in parameters of base class method.")
                            return False
                        
                        # If param is in base class method params
                        cls_param_types = cls_dict[method_name].__annotations__[parameter]

                        # If class params are Union
                        if len(get_args(cls_param_types)) != 0:
                            # print(f"Class parameter {parameter} of method {method_name} are Union with types: {[p for p in get_args(cls_param_types)]}")

                            # If subclass params are Union
                            if len(get_args(sub_param_type)) != 0:
                                # print(f"Subclass parameter {parameter} of method {method_name} type is Union")
                                for sub_param in get_args(sub_param_type):
                                    equality_check = [type(sub_param) == cls_param for cls_param in get_args(cls_param_types)]
                                    sub_check = [issubclass(sub_param, cls_param) for cls_param in get_args(cls_param_types)]

                                    if True not in equality_check:
                                        if True not in sub_check:
                                            return False
                                        
                            # Subclass params aren't Union
                            else:
                                # print(f"Subclass parameter {parameter} of method {method_name} is not Union")
                                equality_check = [type(sub_param_type) == cls_param for cls_param in get_args(cls_param_types)]
                                sub_check = [issubclass(sub_param_type, cls_param) for cls_param in get_args(cls_param_types)]
                                # print(f"\t Equality and subclass check:")
                                # print('\t', equality_check)
                                # print('\t', sub_check)
                                
                                # If subclass param not equal to any type in the class param Union
                                if True not in equality_check:
                                    # If subclass param not a subclass of any type in the class param Union
                                    if True not in sub_check:
                                        return False
                                    
                        # If class param is not Union
                        else:
                            # print(f"Class parameter {parameter} of method {method_name} is not Union, with type {cls_param_types}")

                            # If subclass params are Union
                            if len(get_args(sub_param_type)) != 0:
                                # print(f"Subclass parameter {parameter} of method {method_name} is Union")
                                pass
                                        
                            # Subclass params aren't Union
                            else:
                                # print(f"Subclass parameter {parameter} of method {method_name} is not Union")

                                # Return false if subclass param does not equal base class param
                                if sub_param_type != cls_param_types:
                                    return False
                        
            return True

    @abstractmethod
    def create_component(self, component_id: str) -> ComponentInterface.ComponentTypes:
        """Create a Dash Component without any data.
        Args:
            component_id (str): The ID of the web component
            kwargs (Any): Any additional arguments to pass to the component
        Returns:
            Union[dcc.Graph, go.Figure, dcc.Markdown, html.Div]: The Dash Web component
        """
        pass

    @abstractmethod
    def generate_component_figure(self, figure_input: PluginInputType) -> ComponentInterface.ComponentTypes:
        """
        """
        pass
