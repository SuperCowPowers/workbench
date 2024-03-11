!!! tip inline end "SageWorks Plugins"
    The SageWorks toolkit provides a flexible plugin architecture to expand, enhance, or even replace the [Dashboard](../aws_setup/dashboard_stack.md). Make custom UI components, views, and entire pages with the plugin classes described here.

The SageWorks Plugin system allows clients to customize how their AWS Machine Learning Pipeline is displayed, analyzed, and visualized. Our easy to use Python API enables developers to make new [Dash/Plotly](https://plotly.com/) components, data views, and entirely new web pages focused on business use cases.

### Concept Docs
Many classes in SageWorks need additional high-level material that covers class design and illustrates class usage. Here's the Concept Docs for Plugins [SageWorks Plugins](https://docs.google.com/presentation/d/1sByTnZa24lY6d4INRMm7OHmQndIZmLbTxOyTeAJol20/edit?usp=sharing)

## Make a plugin

Each plugin class inherits from the SageWorks PluginInterface class and needs to set two attributes and implement two methods. These requirements are set so that each Plugin will conform to the Sageworks infrastructure; if the required attributes and methods aren’t included in the class definition, errors will be raised during tests and at runtime.

```

from sageworks.web_components.plugin_interface import PluginInterface, PluginType

class MyPlugin(PluginInterface):
    """My Awesome Component"""

    # Initialize the required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS
    
    # Implement the two methods
    def create_component(self, component_id: str) -> ComponentInterface.ComponentTypes:
        < Function logic which creates a Dash Component (Union[dcc.Graph, dash_table.DataTable, dcc.Markdown, html.Div]) >
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def generate_component_figure(self, figure_input: PluginInputType) -> ComponentInterface.FigureTypes:
        < Function logic which creates a figure (go.Figure) 
        return figure
```
  



### Required Attributes

The class variable plugin_type determines what type of plugin the MyPlugin class is. This variable is inspected during plugin loading at runtime in order to load the plugin to the correct artifact page in the Sageworks dashboard. The PluginType class can be DATA_SOURCE, FEATURE_SET, MODEL, or ENDPOINT.

## S3 Bucket Plugins
Offers the most flexibility and fast prototyping. Simple set your config/env for  blah to an S3 Path and SageWorks will load the plugins from S3 directly.

**Helpful Tip**

You can copy files from your local system up to S3 with this handy AWS CLI call

```
 aws s3 cp . s3://my-sageworks/sageworks_plugins \
 --recursive --exclude "*" --include "*.py"
```
 

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with plugins? Want to develop a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
