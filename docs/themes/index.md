# Summary of Theming Mechanisms

## 1. CSS Files
CSS affects **global styling** of the entire application, including:

- Font styles and sizes.
- Layout styles (e.g., margins, padding, spacing).
- Background and text colors for non-Bootstrap or custom elements.
- Borders and shadows.
- Interactive element styling (e.g., hover effects, link colors).
- Dash custom components.

---

## 2. Plotly Templates
Plotly templates control **figure-specific styling** for Plotly charts and graphs:

- Axis styling (e.g., line colors, grid lines, and ticks).
- Color scales for heatmaps, bar charts, and scatter plots.
- Marker and line styles.
- Background colors for the figure and plot areas.
- Font styles inside the figures.

---

## 3. `data-bs-theme`
The `data-bs-theme` attribute affects **Bootstrap-specific components**, primarily from **Dash Bootstrap Components (DBC)**:

- Buttons.
- Tables.
- Forms (e.g., input boxes, checkboxes, radio buttons).
- Cards, modals, and alerts.

---

## Key Differences
| **Aspect**            | **CSS Files**                | **Plotly Templates**       | **`data-bs-theme`**          |
|------------------------|------------------------------|-----------------------------|------------------------------|
| **Scope**             | Global app styling          | Plotly figures             | Bootstrap-specific elements |
| **Dynamic Switching?** | Not easily (reloading needed)| Can update dynamically     | Supports dynamic switching  |

---

## Conclusion
For **Phase 1**:

- **CSS Files**: Use for global look and feel.
- **Plotly Templates**: Use for consistent chart/figure themes.
- **`data-bs-theme`**: Use to control Bootstrap component styling.




# Theming Strategy for SageWorks Dashboard

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
### Theme Support in SageWorks
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
