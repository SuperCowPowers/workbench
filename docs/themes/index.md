# Theme Support for Workbench
!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

Workbench interfaces are going to have full support for theming and branding to enable customizations of components, and pages. The theming support will include all of the default pages/components but will also support full theming for plugin components and pages [Workbench Plugins](../plugins/index.md)

# Summary of Theming Mechanisms

## 1. CSS Files
CSS affects **global styling** of the entire application, including:

- Font styles and sizes.
- Layout styles (e.g., margins, padding, spacing).
- Background and text colors for non-Bootstrap or custom elements.
- Borders and shadows.
- Interactive element styling (e.g., hover effects, link colors).
- Dash custom components.


## 2. Plotly Figure Templates
Plotly templates control **figure-specific styling** for Plotly charts and graphs:

- Axis styling (e.g., line colors, grid lines, and ticks).
- Color scales for heatmaps, bar charts, and scatter plots.*
- Marker and line styles.
- Background colors for the figure and plot areas.
- Font styles inside the figures.


## 3. DBC Components Light/Dark
The `data-bs-theme=light/dark` html attribute affects **Bootstrap-specific components**, primarily from **Dash Bootstrap Components (DBC)**:

- Buttons.
- Tables.
- Forms (e.g., input boxes, checkboxes, radio buttons).
- Cards, modals, and alerts.



## Key Differences

| <strong>Aspect</strong>            | <strong>CSS Files</strong>                | <strong>Plotly Templates</strong>       | <strong>DBC Dark/Light</strong>          |
|------------------------------------|------------------------------------------|-----------------------------------------|------------------------------------------|
| <strong>Scope</strong>             | Global app styling                      | Plotly figures                          | DBC Components                           |
| <strong>Level of Control</strong>  | Full Range                              | Partial/Full*                           | Just Light and Dark                      |
| <strong>Dynamic Switching?</strong>| ✅ Yes (stylesheet swap)                 | ✅ Yes (plugin callbacks)               | ✅ Yes (`data-bs-theme`)                 |

* Some figure parameters will automatically work, but for stuff like colorscales the component code needs to 'pull' that from the template meta data when it updates it's figure.

## Dynamic Theme Switching (No Page Reload!)

Workbench supports **instant theme switching** without page reload. When you select a new theme from the settings menu, all styling systems update immediately:

- Bootstrap components re-style via `data-bs-theme` attribute
- CSS stylesheets swap dynamically (including custom.css)
- Plotly figures re-render with the new template colors

### How It Works

Theme switching is handled entirely client-side using Dash clientside callbacks:

1. **User clicks a theme** in the settings menu
2. **JavaScript executes** to update localStorage, cookies, and DOM
3. **Bootstrap stylesheet** is swapped by updating the `<link>` href
4. **`data-bs-theme`** attribute is updated (light/dark)
5. **`workbench-theme-store`** is updated, triggering plugin callbacks
6. **Plugins re-render** their figures with new theme colors

### Theme Persistence

Your theme choice is persisted across browser sessions:

- **localStorage**: Stores the theme name for JavaScript access
- **Cookie (`wb_theme`)**: Synced to server for Flask/Plotly template selection

### Three Styling Systems (All Dynamic!)

| **System**           | **What It Styles**              | **How It Switches**                        |
|----------------------|---------------------------------|--------------------------------------------|
| CSS/Bootstrap        | Layout, buttons, cards, tables  | ✅ `data-bs-theme` + stylesheet swap        |
| Plotly Templates     | Figure axes, colors, fonts      | ✅ Plugin callbacks re-render figures       |
| Custom CSS           | Non-Bootstrap elements          | ✅ Cache-busting stylesheet reload          |

### Plugin Theme Support

Plugins can listen for theme changes by using `THEME_STORE_ID` as a callback Input:

```python
from workbench.web_interface.components.plugin_interface import (
    PluginInterface, THEME_STORE_ID
)
from dash import callback, Output, Input

class MyPlugin(PluginInterface):
    def register_internal_callbacks(self):
        @callback(
            Output(self.component_id, "figure", allow_duplicate=True),
            Input(THEME_STORE_ID, "data"),
            prevent_initial_call=True,
        )
        def _update_on_theme_change(theme):
            if self.model is None:
                return self.display_text("Waiting for Data...")
            # Re-render with updated theme colors
            return self.update_properties(self.model)[0]
```

The `PluginInterface` base class provides a shared `theme_manager` instance for accessing colors and colorscales.

### Currently Supported Plugins

The following plugins automatically re-render on theme change:

- ✅ ScatterPlot
- ✅ ConfusionMatrix
- ✅ ShapSummaryPlot

### Future Work

- Add theme switching support to remaining plugins
- Fix checkmark not showing on initial page load (minor UI issue)

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with plugins? Want to develop a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)

