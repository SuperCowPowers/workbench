"""Deploy endpoints for Atom-JEPA models that are already trained and registered.

`cyp_atomjepa.py` trains and deploys in one loop, so a GPU capacity miss at deploy time
strands a trained model with no endpoint. This picks those up without retraining.

    ml_pipeline_launcher cyp_atomjepa_deploy.py -- --models cyp-reg-atomjepa-2c9-readout
"""

import argparse

from workbench.api import Model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", nargs="+", required=True, help="Registered Atom-JEPA models to deploy")
parser.add_argument("--instance", default=None, help="Pin one instance type instead of the Atom-JEPA instance pools")
args = parser.parse_args()

for name in args.models:
    model = Model(name)
    print(f"Deploying {name}...")
    end = model.to_endpoint(tags=model.get_tags(), async_endpoint=True, instance=args.instance)
    end.set_owner("open_admet_cyp")
    end.test_inference()
    end.cross_fold_inference()
    print(f"{name}: endpoint up, cross-fold inference complete")
