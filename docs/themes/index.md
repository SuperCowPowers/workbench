# Theme Support for Workbench
!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)

Workbench supports full theming and branding for customization of components and pages, including [Workbench Plugins](../plugins/index.md).

## Theming Mechanisms

| **System**           | **What It Styles**              | **How It Switches**                        |
|----------------------|---------------------------------|--------------------------------------------|
| CSS/Bootstrap        | Layout, buttons, cards, tables  | `data-bs-theme` + stylesheet swap          |
| Plotly Templates     | Figure axes, colors, fonts      | Plugin `set_theme()` re-renders figures    |
| Custom CSS           | Non-Bootstrap elements          | Cache-busting stylesheet reload            |

*Note: Colorscales require plugins to pull from template metadata when updating figures.*

## Dynamic Theme Switching

Workbench supports **instant theme switching** without page reload:

1. User clicks a theme in the settings menu
2. JavaScript updates localStorage, cookies, and DOM
3. Bootstrap stylesheet swapped via `<link>` href
4. `data-bs-theme` attribute updated (light/dark)
5. `workbench-theme-store` updated, triggering plugin callbacks
6. Plugins re-render figures with new theme colors

**Theme Persistence**: Stored in localStorage (JS access) and cookie `wb_theme` (server access).

## Plugin Theme Support

Plugins support theme changes by overriding the `set_theme()` method:

```python
from workbench.web_interface.components.plugin_interface import PluginInterface
from dash import no_update

class MyPlugin(PluginInterface):
    def __init__(self):
        self.model = None  # Cache data for re-rendering
        super().__init__()

    def set_theme(self, theme: str) -> list:
        """Re-render when theme changes."""
        if self.model is None:
            return [no_update] * len(self.properties)
        return self.update_properties(self.model)
```

The base `PluginInterface` provides:

- Default `set_theme()` returning `[no_update] * len(self.properties)`
- Shared `theme_manager` for accessing colors and colorscales

**Page-level callback wiring** (in `callbacks.py`):

```python
def setup_theme_callback(plugins):
    @callback(
        [Output(cid, prop, allow_duplicate=True) for p in plugins for cid, prop in p.properties],
        Input("workbench-theme-store", "data"),
        prevent_initial_call=True,
    )
    def on_theme_change(theme):
        all_props = []
        for plugin in plugins:
            all_props.extend(plugin.set_theme(theme))
        return all_props
```

Plugins stay simple—they just implement `set_theme()` without knowing about callbacks or store IDs.

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with plugins? Want to develop a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
