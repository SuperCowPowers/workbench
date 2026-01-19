# Understanding Bootstrap and DBC Theme Hooks in Dash

## Overview
Bootstrap and Dash Bootstrap Components (DBC) provide mechanisms to control themes and styling through **CSS classes**, **CSS variables**, and **data attributes**. These hooks enable consistent theming across applications.

---

## Theme Hooks Overview

### 1. **CSS Classes**
- **Description**: Bootstrap provides pre-defined classes like `container`, `row`, `col`, `btn`, etc., for responsive layouts and component styling.
- **Usage**: DBC components automatically apply many of these classes, and you can also customize them using the `className` property.

### 2. **CSS Variables**
- **Description**: Bootstrap uses CSS variables (e.g., `--bs-body-bg`, `--bs-primary`) to allow flexible theming and styling.
- **How They Work**: These variables dynamically adjust based on the theme (e.g., light or dark) or can be customized via custom stylesheets.
- **Common Variables**:
  - `--bs-body-bg`: Background color.
  - `--bs-body-color`: Text color.
  - `--bs-border-color`: Border color.

### 3. **Data Attributes**
- **Description**: The `data-bs-theme` attribute enables quick switching between light and dark themes.
- **Usage**: Setting `data-bs-theme="light"` or `data-bs-theme="dark"` on a root container (e.g., `<body>` or a main `html.Div`) dynamically adjusts Bootstrap variables globally.

---

## How These Hooks Work Together

1. **Class Names**:
   - Used in DBC components to apply Bootstrap styling.
   - Affects layout, grid, and overall component behavior.

2. **CSS Variables**:
   - Allow fine-grained control over the visual appearance based on the `data-bs-theme`.
   - Ensure consistent styling for text, backgrounds, borders, etc.

3. **Data Attributes**:
   - Act as a high-level toggle for light/dark theming, affecting all components globally.
   - Automatically updates the values of Bootstrap CSS variables.

---

## Notes on Redundancy
- When using `data-bs-theme`, explicitly setting styles (e.g., `backgroundColor`) might be redundant since the Bootstrap CSS variables (e.g., `--bs-body-bg`) automatically adjust based on the selected theme.
- DBC components like `dbc.Container` do not automatically set `data-bs-theme` or specific class names, so you may need to explicitly define these for global theming.

---

## Conclusion
- Use **data attributes** (`data-bs-theme`) for global light/dark switching.
- Leverage **CSS variables** (`--bs-*`) for consistent theming across components.
- Apply **CSS classes** (`className`) for additional customization or integration with custom styles.

# Theming Strategy for Workbench Dashboard

## **How Theme Switching Works**

Workbench supports dynamic theme switching at runtime. When a user changes themes, the page reloads to ensure all styling systems are applied consistently.

### **Why We Use Page Reload**

1. **CSS and Plotly Templates**:
   - Global styles are handled via CSS (including Flask-served `custom.css`) and Plotly templates (via `pio.templates.default`) during app initialization.
   - Plotly figures have their styling "baked in" at render time—they don't respond to CSS variable changes.

2. **The Alternative Would Be Complex**:
   - Without reload, every figure would need an explicit callback to re-render when themes change.
   - With 20+ dynamic figures across plugin pages, wiring up individual callbacks isn't practical.

3. **Reload Is Simple and Reliable**:
   - A page reload ensures all three systems (CSS, Plotly templates, Bootstrap) are applied correctly.
   - Theme switching is typically a one-time choice, not something users toggle frequently.

### **What Happens on Theme Change**

1. User selects a new theme
2. Theme preference is saved (persisted across sessions)
3. Page reloads
4. On reload, the new CSS stylesheet, Plotly template, and Bootstrap theme are all applied consistently

<br><br><br>
<br><br><br>


## Storage
### Theme Support in Workbench
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
