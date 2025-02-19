!!! tip inline end "Workbench Plugins"
    The Workbench toolkit provides a flexible plugin architecture to expand, enhance, or even replace the [Dashboard](../aws_setup/dashboard_stack.md). Make custom UI components, views, and entire pages with the plugin classes described here.

The Workbench Plugin system allows clients to customize how their AWS Machine Learning Pipeline is displayed, analyzed, and visualized. Our easy to use Python API enables developers to make new [Dash/Plotly](https://plotly.com/) components, data views, and entirely new web pages focused on business use cases.

### Concept Docs
Many classes in Workbench need additional high-level material that covers class design and illustrates class usage. Here's the Concept Docs for Plugins:

- [Workbench Plugin Overview](https://docs.google.com/presentation/d/1RjpMmJW1i9auPztn2xXYmYKXsZjsnG7vVaCQQ4FLIMM/edit?usp=sharing)


## Make a plugin

Each plugin class inherits from the Workbench PluginInterface class and needs to set two attributes and implement two methods. These requirements are set so that each Plugin will conform to the Workbench infrastructure; if the required attributes and methods aren’t included in the class definition, errors will be raised during tests and at runtime.

**Note:** For full code see [Model Plugin Example](https://github.com/SuperCowPowers/workbench/blob/main/examples/plugins/components/model_plugin.py)

```

class ModelPlugin(PluginInterface):
    """MyModelPlugin Component"""

    """Initialize this Plugin Component """
    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create the container for this component
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        self.component_id = component_id
        self.container = dcc.Graph(id=component_id, ...)

        # Fill in plugin properties
        self.properties = [(self.component_id, "figure")]

        # Return the container
        return self.container

    def update_properties(self, model: Model, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            model (Model): An instantiated Model object
            **kwargs: Additional keyword arguments

        Returns:
            list: A list of the updated property values
        """

        # Create a pie chart with the endpoint name as the title
        pie_figure = go.Figure(data=..., ...)

        # Return the updated property values for the plugin
        return [pie_figure]

```
  


### Required Attributes

The class variable plugin\_page determines what type of plugin the MyPlugin class is. This variable is inspected during plugin loading at runtime in order to load the plugin to the correct artifact page in the Workbench dashboard. The PluginPage class can be DATA_SOURCE, FEATURE\_SET, MODEL, or ENDPOINT.

## S3 Bucket Plugins
Offers the most flexibility and fast prototyping. Simply set your workbench  S3 Path and Workbench will load the plugins from S3 directly.

```
"WORKBENCH_PLUGINS": "s3://my-s3-bucket/workbench_plugins"
```

**Helpful Tip**

You can copy files from your local system up to S3 with this handy AWS CLI call

```
 aws s3 cp . s3://my-s3-bucket/workbench_plugins \
 --recursive --exclude "*" --include "*.py"
```
 

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with plugins? Want to develop a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
