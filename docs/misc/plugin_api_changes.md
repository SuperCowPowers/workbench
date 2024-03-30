## Plugin API Changes
There were quite a fiew API changes for Plugins between `0.4.43` and `0.5.0` versions of SageWorks.

* Classes that inherit from `component_interface` or `plugin_interface`  are now 'auto wrapped' with an exception container. This container not only catches errors/crashes so they don't crash the application but it also displays the error in the widget.
* The `generate_component_figure` method has changed to `update_contents`
* `PluginType` was changed to `PluginPage` (use CUSTOM to NOT autoload)
* `PluginInputType.MODEL_DETAILS`  changed to `PluginInputType.MODEL`  (since your now getting a model object)