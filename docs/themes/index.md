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
| <strong>Dynamic Switching?</strong>| Not easily (reloading needed)           | Can update dynamically                  | Supports dynamic switching               |

* Some figure parameters will automatically work, but for stuff like colorscales the component code needs to 'pull' that from the template meta data when it updates it's figure.

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with plugins? Want to develop a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)

