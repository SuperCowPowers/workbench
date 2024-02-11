!!! tip inline end "SageWorks Plugins"
    The SageWorks toolkit provides a flexible plugin architecture to expand, enhance, or even replace the [Dashboard](../aws_setup/dashboard_stack.md). Make custom UI components, views, and entire pages with the plugin classes described here.

The SageWorks Plugin system allows clients to customize how their AWS Machine Learning Pipeline is displayed, analyzed, and visualized. Our easy to use Python API enables developers to make new [Dash/Plotly](https://plotly.com/) components, data views, and entirely new web pages focused on business use cases.

### Concept Docs
Many classes in SageWorks need additional high-level material that covers class design and illustrates class usage. Here's the Concept Docs for Plugins [SageWorks Plugins](https://docs.google.com/presentation/d/1sByTnZa24lY6d4INRMm7OHmQndIZmLbTxOyTeAJol20/edit?usp=sharing)

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
