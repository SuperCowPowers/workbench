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
        default="molecule_name",
        help="Name of the ID column (default: molecule_name)",
    )
    args = parser.parse_args()
    models = args.models
    id_column = args.id_column

    # Create MetaModelSimulator instance and generate report
    sim = MetaModelSimulator(models, id_column=id_column)
    sim.report()


if __name__ == "__main__":
    main()
