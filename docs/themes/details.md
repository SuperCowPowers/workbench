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

## **Phase 1: Application Start-Up Theming**
This focuses on setting a polished, cohesive interface at startup. It ensures:

1. **CSS and Plotly Templates at Start-Up**:
   - The current system handles global styles via CSS (including Flask-served `custom.css`) and Plotly templates (via `pio.templates.default`) during app initialization.
2. **Highly Customizable Interface**:
   - By leveraging both custom CSS and Plotly templates, you can style everything to match branding and preferences.

### **Why Phase 1 Works Well**
- **Plotly Templates**: Figures are automatically styled if `pio.templates.default` is set before app initialization.
- **CSS**: Dynamically affects the entire app without additional callbacks.
- **Scalability**: Suitable for large apps since styles and templates are applied globally.

<br><br>

## **Phase 2: Dynamic Theme Switching**
Dynamic theme switching allows users to toggle between themes (e.g., light and dark) at runtime. This is more complex, and splitting it into its own phase will reduce the complexity of each Phase.

### **Challenges with Dynamic Switching**
1. **Encapsulation Barriers**:
   - A centralized callback for theme switching needs to know about every figure or component in the app, which is not scalable.
2. **Dynamic Content/Plugins**:
   - We have plugin pages, views, and components. A Typical app has 20 figures dynamically created, adding complexity to centralized callbacks.



### **Proposed Solution: `set_theme()` Method**
Introduce a `set_theme()` method in each component to manage its own updates:

- **Figures**: Update their `layout.template` or regenerate themselves.
- **HTML/CSS Components**: Dynamically update styles or toggle CSS classes.

### **Advantages of `set_theme()`**
1. **Encapsulation**:
   - Each component knows how to update itself based on the theme.
2. **Simple Callback**:
   - The theme switch callback iterates through all registered components and calls their `set_theme()` method.
3. **Scalable**:
   - Suitable for multi-page apps with many dynamic components.



## **How the Two Phases Work Together**
### Phase 1:
- **Focus**: Global theming at app start-up.
- **What Works**:
  - Plotly templates are applied to all figures by default.
  - CSS styles are applied across the app.

### Phase 2:
- **Focus**: Extend components to dynamically update their theme.
- **What to Add**:
  - Extend components with a `set_theme(theme_name: str)` method.
  - The callback triggers `set_theme()` for each registered component.



## **Next Steps**
- **Phase 1**: Finalize CSS and Plotly template integration for start-up theming.
- **Phase 2**: Begin implementing a `set_theme()` system for individual components.

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
