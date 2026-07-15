"""ModelComparisonView: data assembly for the Model Comparison plugin page.

Pulls the champion/challenger contests from CachedMeta and shapes them into the
DataFrames the page renders: champion tables (regression/classification split),
ranked challenger tables, and per-pair comparison metrics and predictions.
"""

import pandas as pd

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_model import CachedModel
from workbench.core.artifacts.model_core import ModelType
from workbench.utils.model_comparison import (
    contest_ranking,
    model_comparison,
    prediction_comparison,
    rank_models,
)


class ModelComparisonView(PageView):
    def __init__(self, inference_run: str = "full_cross_fold"):
        """ModelComparisonView: champion/challenger contest data for the comparison page"""
        super().__init__()
        self.meta = CachedMeta()
        self.inference_run = inference_run

        # Champion tables (regression/classification split) and endpoint -> champion map
        self.reg_champions = pd.DataFrame()
        self.class_champions = pd.DataFrame()
        self.endpoint_champion = {}
        self.refresh()

    def refresh(self):
        """Refresh the champion tables from CachedMeta"""
        champs = self.meta.champion_models()
        self.endpoint_champion = dict(zip(champs["Endpoint"], champs["Model"]))

        # Split champions by model type (classifiers right, everything else left)
        endpoint_map = dict(zip(champs["Model"], champs["Endpoint"]))
        reg_models, class_models = [], []
        for name in champs["Model"]:
            model = CachedModel(name)
            (class_models if model.model_type == ModelType.CLASSIFIER else reg_models).append(model)
        self.reg_champions = self._champion_table(reg_models, endpoint_map)
        self.class_champions = self._champion_table(class_models, endpoint_map)

    def _champion_table(self, models: list, endpoint_map: dict) -> pd.DataFrame:
        """Champion table: Model, Endpoint, and the ranked metrics columns.
        Champions without metrics for the run still get a row (blank metrics, ranked last)."""
        if not models:
            return pd.DataFrame(columns=["Model", "Endpoint"])
        table = rank_models(models, self.inference_run)
        no_metrics = [m.name for m in models if m.name not in table.index]
        table = table.reindex(list(table.index) + no_metrics).round(3)
        table.insert(0, "Model", table.index)
        table.insert(1, "Endpoint", table["Model"].map(endpoint_map))
        return table.reset_index(drop=True)

    def challengers(self, endpoint: str) -> pd.DataFrame:
        """Ranked challenger table (with Δ-vs-champion columns) for a champion endpoint"""
        champion = CachedModel(self.endpoint_champion[endpoint])
        challengers = [CachedModel(name) for name in self.meta.challenger_models(endpoint)]
        ranked = contest_ranking(champion, challengers, self.inference_run)
        if ranked is None or ranked.empty:
            return pd.DataFrame(columns=["Model"])
        ranked = ranked.round(3)
        ranked.insert(0, "Model", ranked.index)
        return ranked.reset_index(drop=True)

    def comparison(self, champion: str, challenger: str) -> pd.DataFrame:
        """The champion/challenger/delta metrics table for a selected pair"""
        comp = model_comparison(CachedModel(champion), CachedModel(challenger), self.inference_run)
        if comp is None:
            return pd.DataFrame(columns=["Model"])
        comp = comp.round(3)
        comp.insert(0, "Model", comp.index)
        return comp.reset_index(drop=True)

    def plot_data(self, champion: str, challenger: str) -> pd.DataFrame:
        """Stacked predictions for the ComparisonPlot"""
        return prediction_comparison(CachedModel(champion), CachedModel(challenger), self.inference_run)


if __name__ == "__main__":
    # Exercising the ModelComparisonView
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    view = ModelComparisonView()

    print("*** Regression Champions ***")
    print(view.reg_champions)

    print("\n*** Classification Champions ***")
    print(view.class_champions)

    endpoint = view.reg_champions["Endpoint"].iloc[0]
    print(f"\n*** Challengers for '{endpoint}' ***")
    challengers = view.challengers(endpoint)
    print(challengers)

    champion = view.endpoint_champion[endpoint]
    challenger = challengers["Model"].iloc[0]
    print(f"\n*** Comparison: {champion} vs {challenger} ***")
    print(view.comparison(champion, challenger))
