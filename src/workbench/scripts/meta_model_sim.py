"""MetaModelSimulator: Simulate and analyze ensemble model performance.

This class helps evaluate whether a meta model (ensemble) would outperform
individual child models by analyzing endpoint inference predictions.
"""

import argparse
from workbench.utils.meta_model_simulator import MetaModelSimulator


def main():
    parser = argparse.ArgumentParser(
        description="Simulate and analyze ensemble model performance using MetaModelSimulator."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="List of model endpoint names to include in the ensemble simulation.",
    )
    parser.add_argument(
        "--id-column",
        required=True,
        help="Name of the ID column used for row alignment across models.",
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
    sim = MetaModelSimulator(args.models, id_column=args.id_column, capture_name=args.capture_name)
    sim.report()
    if args.output:
        df = sim.best_ensemble_predictions()
        df.to_csv(args.output, index=False)
        print(f"\nEnsemble predictions saved to {args.output}")


if __name__ == "__main__":
    main()
