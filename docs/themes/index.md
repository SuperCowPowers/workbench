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

## Why We Require a Page Reload for Theme Switching

Workbench supports dynamic theme switching—you can change themes at runtime. However, switching themes triggers a page reload rather than an instant in-place update. If you've seen demos like the [Dash Bootstrap Theme Explorer Gallery](https://hellodash.pythonanywhere.com/theme-explorer/gallery) where themes switch instantly, you might wonder why we can't do the same.

### The Technical Reality

**Plotly figures don't respond to CSS variables.** Unlike Bootstrap components (buttons, cards, tables) which automatically re-style when `data-bs-theme` changes, Plotly figures have their styling "baked in" at render time. The figure's colors, fonts, and backgrounds are set when the figure is created—they don't dynamically respond to theme changes.

### What dash-bootstrap-templates Actually Does

Even the excellent [dash-bootstrap-templates](https://github.com/AnnMarieW/dash-bootstrap-templates) library has this limitation. From their docs:

> *"The All-in-One component switches the Bootstrap stylesheet and sets the default Plotly figure template, however, figures must be updated in a callback in order to render the figure with the new template."*

This means every figure needs an explicit callback to re-render when the theme changes. In a simple demo app with 3-5 figures, that's manageable. In Workbench, with 20+ dynamic figures across multiple plugin pages, wiring up individual callbacks for each figure isn't practical.

### Three Separate Styling Systems

| **System**           | **What It Styles**              | **Instant Switching?**                     |
|----------------------|---------------------------------|--------------------------------------------|
| CSS/Bootstrap        | Layout, buttons, cards, tables  | ✅ Yes (via `data-bs-theme`)                |
| Plotly Templates     | Figure axes, colors, fonts      | ❌ No - requires figure re-render           |
| Custom CSS           | Non-Bootstrap elements          | ⚠️ Requires stylesheet swap                 |

### Our Approach: Reload on Theme Change

Rather than implementing complex callback wiring for every figure across all plugin pages, Workbench takes a pragmatic approach: **when you switch themes, the page reloads**. This ensures all three styling systems (CSS, Plotly templates, and Bootstrap components) are applied consistently and correctly.

The reload is fast, and theme selection is typically a one-time choice rather than something users toggle frequently. This approach gives us full theme flexibility without the architectural complexity of in-place figure updates.

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with plugins? Want to develop a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)

