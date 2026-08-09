"""PredictionPlot: how a model did on its inference set, with the chrome removed.

Workbench's ModelPlot already picks the right visualization per model type -- a scatter of
predictions vs actuals for regressors, a confusion view for classifiers -- so this subclasses
it rather than reimplementing the switch, and just swaps the two sub-plugins for quieter ones:

  * ScatterPlot(show_controls=False): drops the X/Y/Color dropdowns and the Diagonal and
    Prediction Interval checkboxes. The diagonal LINE still draws (ModelPlot passes
    regression_line=True), which is the part worth keeping on an actual-vs-predicted plot.
  * ConfusionMatrix: just the matrix, instead of the ConfusionExplorer's matrix + triangle.

Both get a shorter height and scroll_zoom=False, so on this tall page the mouse wheel scrolls
the page instead of zooming the embedded plot. Everything else -- visibility switching,
multi-task target handling, theme re-rendering -- is inherited from ModelPlot.
"""

from workbench.cached.cached_model import CachedModel
from workbench.web_interface.components.plugins.confusion_matrix import ConfusionMatrix
from workbench.web_interface.components.plugins.model_plot import ModelPlot
from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

# Champion and challenger rows stack on this page, so keep the plots short enough that a
# comparison fits without scrolling off the bottom.
PLOT_HEIGHT = 300

# The confusion matrix is a bold square that dominates at full height, so give it less
# than the scatter to calm it down.
CONFUSION_HEIGHT = 250


class PredictionPlot(ModelPlot):
    """A quieter ModelPlot: no scatter controls, plain confusion matrix, no scroll-zoom."""

    def __init__(self):
        """Initialize the PredictionPlot plugin class"""
        super().__init__()

        # ModelPlot only ever touches these two generically (create_component, .properties,
        # .signals, update_properties, register_internal_callbacks), so swapping the instances
        # is enough -- no need to reimplement any of its logic. Inheritance satisfies the
        # PluginInterface contract, so no pass-through create_component/update_properties needed.
        #
        # Note the inherited attribute name: confusion_explorer holds a plain ConfusionMatrix
        # here. Keeping ModelPlot's name is what lets the rest of it work untouched.
        self.scatter_plot = ScatterPlot(show_controls=False, height=PLOT_HEIGHT, scroll_zoom=False)
        self.confusion_explorer = ConfusionMatrix(height=CONFUSION_HEIGHT, scroll_zoom=False)


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # A regressor shows the scatter (no controls); swap for a classifier to see the matrix.
    model = CachedModel("aqsol-regression")
    PluginUnitTest(PredictionPlot, input_data=model, theme="dark").run()
