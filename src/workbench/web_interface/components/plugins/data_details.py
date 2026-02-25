"""A Markdown Component for details/information about DataSources (and FeatureSets)"""

from typing import Union
from html import escape
import logging

from dash import html, dcc

from workbench.api import DataSource, FeatureSet
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.markdown_utils import tags_to_markdown

log = logging.getLogger("workbench")

NUMERIC_TYPES = {"tinyint", "smallint", "int", "bigint", "float", "double", "decimal"}
FLOAT_TYPES = {"float", "double", "decimal"}
SKIP_TYPES = NUMERIC_TYPES | {"timestamp"}
FS_TYPE_MAP = {"Integral": "I", "Fractional": "F", "String": "S", "Timestamp": "TS", "Boolean": "B"}


class DataDetails(PluginInterface):
    """DataSource/FeatureSet Details Component"""

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATA_SOURCE

    def __init__(self):
        """Initialize the DataDetails plugin class"""
        self.component_id = None
        self.data_source = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for the Model Details
        """
        self.component_id = component_id
        self.properties = [
            (f"{component_id}-header", "children"),
            (f"{component_id}-details", "children"),
        ]
        return html.Div(
            id=component_id,
            children=[
                html.H3(id=f"{component_id}-header", children="Loading..."),
                dcc.Markdown(id=f"{component_id}-details", dangerously_allow_html=True),
            ],
        )

    def update_properties(self, artifact: Union[DataSource, FeatureSet], **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            artifact (Union[DataSource, FeatureSet]): An instantiated DataSource or FeatureSet
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        log.important(f"Updating Plugin with Artifact: {artifact.name} and kwargs: {kwargs}")
        self.data_source = artifact.data_source if isinstance(artifact, FeatureSet) else artifact
        return [self.data_source.name, self.data_details_markdown(self.data_source.details())]

    def data_details_markdown(self, data_details: dict) -> str:
        if not data_details:
            return "No data source details found"

        # Clean up date formatting
        for key in ("created", "modified"):
            val = data_details.get(key, "")
            if isinstance(val, str) and ".000Z" in val:
                data_details[key] = val.replace(".000Z", "").replace("T", " ")

        tags = tags_to_markdown(data_details.get("workbench_tags", []))
        column_stats = data_details.get("column_stats", {})

        return (
            f"**Rows:** {data_details.get('num_rows', 'N/A')}  \n"
            f"**Columns:** {data_details.get('num_columns', 'N/A')}  \n"
            f"**Created/Mod:** {data_details.get('created', 'N/A')} / {data_details.get('modified', 'N/A')}  \n"
            f"{tags}  \n"
            f"**Input:** {data_details.get('input', 'N/A')}  \n"
            f"<br>  \n"
            f"\n#### Numeric Columns\n\n"
            f"{self._build_numeric_details(column_stats)}\n"
            f"\n<div style='margin-top: 15px;'></div>\n\n"
            f"#### Non-Numeric Columns\n\n"
            f"{self._build_string_details(column_stats)}\n"
        )

    def _build_numeric_details(self, column_stats: dict) -> str:
        """Build HTML details for numeric columns."""
        details = []
        for col_name, col_info in column_stats.items():
            if col_info["dtype"] not in NUMERIC_TYPES:
                continue

            items = [f"<li>{q}: {v:.3f}</li>" for q, v in col_info["descriptive_stats"].items() if v is not None]
            items.append(f"<li>Unique: {col_info['unique']}</li>")

            if col_info.get("correlations"):
                corr_items = "".join(f"<li>{escape(str(c))}: {v:.3f}</li>" for c, v in col_info["correlations"].items())
                corr_title = '<span class="green-text"><b>Correlated Columns</b></span>'
                items.append(f'<li class="no-bullet">{self._expanding_list(corr_title, corr_items)}</li>')

            details.append(self._expanding_list(self._column_summary_html(col_name, col_info), "".join(items)))

        return "\n".join(details)

    def _build_string_details(self, column_stats: dict) -> str:
        """Build HTML details for non-numeric/non-timestamp columns."""
        details = []
        for col_name, col_info in column_stats.items():
            if col_info["dtype"] in SKIP_TYPES:
                continue

            if "value_counts" not in col_info:
                items = "<li>No Value Counts</li>"
            else:
                sorted_counts = sorted(col_info["value_counts"].items(), key=lambda x: x[1], reverse=True)
                items = "".join(f"<li>{escape(str(v))}: {c}</li>" for v, c in sorted_counts)

            details.append(self._expanding_list(self._column_summary_html(col_name, col_info), items))

        return "\n".join(details)

    @staticmethod
    def _expanding_list(summary_html: str, bullet_html: str) -> str:
        """Wrap content in an HTML <details> collapsible section."""
        return f"<details>\n  <summary>{summary_html}</summary>\n  <ul>{bullet_html}</ul>\n</details>"

    def _column_summary_html(self, column_name: str, column_info: dict) -> str:
        """Create a summary HTML string for a column."""
        if "fs_dtype" in column_info:
            display_type = f"{FS_TYPE_MAP.get(column_info['fs_dtype'], 'V')}: {column_info['dtype']}"
        else:
            display_type = column_info["dtype"]

        parts = [f'<b>{escape(column_name)}</b> <span class="blue-text">({escape(display_type)})</span>:']

        dtype = column_info["dtype"]
        if dtype in NUMERIC_TYPES:
            min_val = column_info["descriptive_stats"]["min"]
            max_val = column_info["descriptive_stats"]["max"]

            if min_val is None or max_val is None:
                parts.append(' <span class="red-text">No Stats</span>')
            elif dtype in FLOAT_TYPES:
                parts.append(f" {min_val:.2f} → {max_val:.2f}&nbsp;&nbsp;&nbsp;&nbsp;")
            else:
                parts.append(f" {int(min_val)} → {int(max_val)}&nbsp;&nbsp;&nbsp;&nbsp;")

            if column_info["unique"] == 2 and min_val == 0 and max_val == 1:
                parts.append(' <span class="green-text"> Binary</span>')
            elif column_info["num_zeros"] > 0:
                parts.append(f' <span class="orange-text"> Zero: {column_info["num_zeros"]}</span>')
        else:
            parts.append(f' Unique: {column_info["unique"]} ')

        if column_info["nulls"] > 0:
            parts.append(f' <span class="red-text">Null: {column_info["nulls"]}</span>')

        return "".join(parts)


if __name__ == "__main__":
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    PluginUnitTest(DataDetails).run()
