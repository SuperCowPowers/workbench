"""SettingsMenu: A settings menu component for the Workbench Dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc

# Workbench Imports
from workbench.utils.theme_manager import ThemeManager


class SettingsMenu:
    """A settings menu with admin links and theme selection."""

    def __init__(self):
        """Initialize the SettingsMenu."""
        self.tm = ThemeManager()

    def create_component(self, component_id: str) -> html.Div:
        """Create a settings menu dropdown component.

        Args:
            component_id (str): The ID prefix for the component.

        Returns:
            html.Div: A Div containing the settings menu dropdown.
        """
        themes = self.tm.list_themes()

        # Create theme submenu items
        theme_items = []
        for theme in sorted(themes):
            theme_items.append(
                dbc.DropdownMenuItem(
                    [
                        html.Span(
                            "",
                            id={"type": f"{component_id}-checkmark", "theme": theme},
                            style={
                                "fontFamily": "monospace",
                                "marginRight": "5px",
                                "width": "20px",
                                "display": "inline-block",
                            },
                        ),
                        theme.replace("_", " ").title(),
                    ],
                    id={"type": f"{component_id}-theme-item", "theme": theme},
                )
            )

        # Hamburger icon (3 rounded lines)
        hamburger_icon = html.Div(
            [
                html.Div(
                    style={
                        "width": "20px",
                        "height": "3px",
                        "backgroundColor": "currentColor",
                        "borderRadius": "2px",
                        "marginBottom": "4px",
                    }
                ),
                html.Div(
                    style={
                        "width": "20px",
                        "height": "3px",
                        "backgroundColor": "currentColor",
                        "borderRadius": "2px",
                        "marginBottom": "4px",
                    }
                ),
                html.Div(
                    style={"width": "20px", "height": "3px", "backgroundColor": "currentColor", "borderRadius": "2px"}
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center"},
        )

        # Build menu items: Home, Status, License, divider, Themes submenu
        menu_items = [
            dbc.DropdownMenuItem("Home", href="/"),
            dbc.DropdownMenuItem("Status", href="/status", external_link=True, target="_blank"),
            dbc.DropdownMenuItem("License", href="/license", external_link=True, target="_blank"),
            dbc.DropdownMenuItem(divider=True),
            dbc.DropdownMenuItem("Themes", header=True),
            *theme_items,
        ]

        return html.Div(
            [
                dbc.DropdownMenu(
                    label=hamburger_icon,
                    children=menu_items,
                    id=f"{component_id}-dropdown",
                    toggle_style={
                        "background": "transparent",
                        "border": "none",
                        "boxShadow": "none",
                        "padding": "5px 10px",
                    },
                    caret=False,
                    align_end=True,
                ),
                # Store to trigger checkmark update on load and theme change
                dcc.Store(id=f"{component_id}-init", data=True),
            ],
            id=component_id,
        )

    def get_clientside_callback_code(self) -> str:
        """Get the JavaScript code for the theme selection clientside callback.

        Returns:
            str: JavaScript code for the clientside callback.
        """
        return self.tm.get_theme_switch_js()

    @staticmethod
    def get_checkmark_callback_code() -> str:
        """Get the JavaScript code to update checkmarks based on current theme.

        Returns:
            str: JavaScript code for the checkmark update callback.
        """
        return """
        function(theme, ids) {
            // If theme is a string (from theme switch), use it directly
            let currentTheme = (typeof theme === 'string') ? theme : null;

            // Otherwise, get from localStorage
            if (!currentTheme) {
                currentTheme = localStorage.getItem('wb_theme');
            }

            // Return checkmarks for each theme
            return ids.map(id => id.theme === currentTheme ? '\u2713' : '');
        }
        """


if __name__ == "__main__":
    # Quick test to verify component creation
    menu = SettingsMenu()
    component = menu.create_component("test-settings-menu")
    print("SettingsMenu component created successfully")
    print(f"Available themes: {menu.tm.list_themes()}")
    print(f"Current theme: {menu.tm.current_theme()}")
