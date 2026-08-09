"""A Dash app that stamps the active Workbench theme onto the page."""

import logging

from dash import Dash

# Workbench Imports
from workbench.utils.theme_manager import ThemeManager

log = logging.getLogger("workbench")


class ThemedDash(Dash):
    """Dash app that sets `data-bs-theme` on <html> for every request.

    Bootstrap scopes its whole palette to `[data-bs-theme]`, and the bootswatch builds only
    define a coherent dark set inside that block -- their bare `:root` pairs dark backgrounds
    with light borders and surfaces. The attribute belongs on <html>: page CSS selects it as
    `:root[data-bs-theme=...]`, and setting it on an inner element would override that for
    the element's subtree.
    """

    def interpolate_index(self, **kwargs) -> str:
        """Insert the theme attribute into the rendered index.

        Runs per request, and ThemeManager resolves the theme from that request's cookie, so
        it reflects the viewer's choice rather than whatever the server booted with.
        """
        index = super().interpolate_index(**kwargs)
        if "<html>" not in index:
            log.error("No bare <html> tag in the index template: the theme attribute wasn't applied")
            return index
        return index.replace("<html>", f'<html data-bs-theme="{ThemeManager().data_bs_theme()}">', 1)
