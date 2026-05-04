"""EnsembleSimulator CLI: simulate and analyze ensemble performance.

Evaluates whether an ensemble of models outperforms the individual
children by analyzing their captured cross-fold inference predictions.

Used standalone for offline what-if analysis. The same simulator is also
called automatically by ``MetaEndpoint.create()`` when a DAG contains a
tunable aggregation node.
"""

import argparse

from workbench.api import Endpoint, FeatureSet, Model
from workbench.utils.ensemble_simulator import EnsembleSimulator


def _resolve_lineage(endpoint_name: str) -> tuple[str, str]:
    """Backtrace endpoint → model → FeatureSet → id_column.

    Returns ``(model_name, id_column)`` for the simulator constructor.
    """
    ep = Endpoint(endpoint_name)
    model = Model(ep.get_input())
    fs = FeatureSet(model.get_input())
    return model.name, fs.id_column


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate and analyze ensemble model performance.")
    parser.add_argument(
        "endpoints",
        nargs="+",
        help="List of endpoint names to include in the ensemble simulation.",
    )
    parser.add_argument(
        "--capture-name",
        default="full_cross_fold",
        help="Inference capture name to load predictions from (default: full_cross_fold)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV file path to save best ensemble predictions.",
    )
    args = parser.parse_args()

    model_names: list[str] = []
    id_column: str | None = None
    for ep_name in args.endpoints:
        model_name, fs_id_column = _resolve_lineage(ep_name)
        model_names.append(model_name)
        if id_column is None:
            id_column = fs_id_column

    sim = EnsembleSimulator(model_names, id_column=id_column, capture_name=args.capture_name)
    sim.report()

    if args.output:
        df = sim.best_ensemble_predictions()
        df.to_csv(args.output, index=False)
        print(f"\nEnsemble predictions saved to {args.output}")


if __name__ == "__main__":
    main()
