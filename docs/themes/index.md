# Theme Support in SageWorks
For a Dash/Plotly web applications there are two main concepts that fall under the general umbrella of 'theme support'.

### CSS for High-Level Theming
- **Purpose**: Styles the overall **web interface** (e.g., layout, buttons, dropdowns, fonts, background colors).
- **Scope**: Applies to **Dash components** (e.g., `dcc.Graph`, `dbc.Button`) and layout elements.
- **Implementation**:
  - Use a CSS file or a framework like Bootstrap.
  - Dynamically switch themes (e.g., light/dark) using CSS class changes.

### Templates for Plotly Figures
- **Purpose**: Styles **Plotly figures** (e.g., background, gridlines, colorscales, font styles) to match the app's theme.
- **Scope**: Only affects Plotly-generated visuals like charts, graphs, and heatmaps.
- **Implementation**:
  - Use predefined templates (e.g., `darkly`, `mintly`) or create custom JSON templates.
  - Apply globally or on a per-figure basis.
- **Great Resource**:
    [dash-bootstrap-templates](https://github.com/AnnMarieW/dash-bootstrap-templates)

### How They Complement Each Other
1. **CSS handles the web app’s overall look**:

    Example: A "dark mode" CSS class changes the app's background, text color, and component styles.

2. **Templates handle Plotly-specific figure styling**:
    
    Example: A "dark mode" Plotly template ensures charts have dark backgrounds, white gridlines, and matching font colors.
