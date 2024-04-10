from dash import callback, Output
from dash.exceptions import PreventUpdate
import logging

# SageWorks Imports
from sageworks.api import Model, Endpoint

log = logging.getLogger("sageworks")


def register_callbacks(plugins, input_sources, object_type):
    # Construct a list of Output objects dynamically based on the plugins' content_slots
    outputs = [Output(component_id, property) for plugin in plugins for component_id, property in plugin.slots]

    @callback(
        outputs,
        input_sources,
        prevent_initial_call=True,
    )
    def update_plugin_contents(*args):
        # Unpack the input arguments
        if object_type == "model":
            inference_run, selected_rows, table_data = args
        else:  # object_type == 'endpoint'
            selected_rows, table_data = args

        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        object_uuid = selected_row_data["uuid"]

        # Instantiate the object (Model or Endpoint)
        if object_type == "model":
            obj = Model(object_uuid, legacy=True)
        else:  # object_type == 'endpoint'
            obj = Endpoint(object_uuid, legacy=True)

        # Update the plugins and collect the updated properties for each slot
        updated_properties = []
        for plugin in plugins:
            log.important(f"Updating Plugin: {plugin} with {object_type.capitalize()}: {object_uuid}")
            if object_type == "model":
                updated_contents = plugin.update_contents(obj, inference_run=inference_run)
            else:  # object_type == 'endpoint'
                updated_contents = plugin.update_contents(obj)

            # Assume that the length of contents matches the number of slots for the plugin
            if len(updated_contents) != len(plugin.slots):
                raise ValueError(
                    f"Plugin {plugin} has {len(updated_contents)} content values != {len(plugin.slots)} slots."
                )

            # Append each value from contents to the updated_properties list
            updated_properties.extend(updated_contents)

        # Return the updated properties for each slot
        return updated_properties
