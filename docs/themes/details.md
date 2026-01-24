# Bootstrap and DBC Theme Hooks in Dash

## Theme Hooks Overview

### CSS Classes
Bootstrap provides pre-defined classes (`container`, `row`, `col`, `btn`, etc.) for responsive layouts. DBC components apply these automatically, and you can customize via the `className` property.

### CSS Variables
Bootstrap uses CSS variables for flexible theming:

- `--bs-body-bg`: Background color
- `--bs-body-color`: Text color
- `--bs-border-color`: Border color

These adjust automatically based on the `data-bs-theme` attribute.

### Data Attributes
The `data-bs-theme` attribute (`light` or `dark`) on a root container dynamically adjusts all Bootstrap variables globally.

---

## How Theme Switching Works

Workbench supports **instant theme switching**—no page reload required.

### What Happens on Theme Change

1. User clicks a theme in the settings menu
2. Clientside callback (JavaScript) executes:
   - Saves to `localStorage` and cookie (`wb_theme`)
   - Swaps Bootstrap stylesheet URL
   - Updates `data-bs-theme` attribute
   - Updates `workbench-theme-store` dcc.Store
3. Page-level callback fires, calling `set_theme()` on each plugin
4. Plugins re-render figures with new colors from ThemeManager

### Server-Side Theme Detection

On page load, the server reads the theme from cookies:

```python
@app.server.before_request
def check_theme_cookie():
    theme_name = request.cookies.get("wb_theme")
    if theme_name and theme_name != cls.current_theme_name:
        cls.set_theme(theme_name)
```

This ensures Plotly templates are set correctly for initial figure rendering.

---

## Two Main Theming Concepts

### CSS for Web Interface
Styles the overall layout, buttons, dropdowns, fonts, and background colors. Use Bootstrap or custom CSS files. Switch themes dynamically via CSS class changes.

### Plotly Templates for Figures
Styles Plotly figures (background, gridlines, colorscales, fonts). Use predefined templates or create custom JSON templates.

**Resource**: [dash-bootstrap-templates](https://github.com/AnnMarieW/dash-bootstrap-templates)
