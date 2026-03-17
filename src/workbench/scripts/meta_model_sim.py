"""MetaModelSimulator: Simulate and analyze ensemble model performance.

CLI tool for evaluating whether a meta model (ensemble) would outperform
individual child models by analyzing endpoint inference predictions.

For creating a MetaModel, use `MetaModel.create()` which auto-simulates internally.
"""

import argparse
from workbench.api.meta_model import MetaModel


def main():
    parser = argparse.ArgumentParser(
        description="Simulate and analyze ensemble model performance."
    )
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

    # Run simulation via MetaModel.simulate() (auto-resolves id_column)
    sim = MetaModel.simulate(args.endpoints, capture_name=args.capture_name)
    sim.report()

    if args.output:
        df = sim.best_ensemble_predictions()
        df.to_csv(args.output, index=False)
        print(f"\nEnsemble predictions saved to {args.output}")


if __name__ == "__main__":
    main()
