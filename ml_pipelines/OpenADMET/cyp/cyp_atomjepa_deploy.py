"""Finish Atom-JEPA models that are registered but not yet serving out-of-fold predictions.

`cyp_atomjepa.py` trains every isoform before deploying any, so a GPU capacity miss leaves a
trained model behind. This picks those up without retraining, and a model whose endpoint is
already up only runs the inference it is missing -- redeploying one would cost another GPU
provision cycle and discard its captures.

    ml_pipeline_launcher cyp_atomjepa_deploy.py -- --models cyp-reg-atomjepa-2d6-readout
"""

import argparse

from workbench.api import Endpoint, Model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", nargs="+", required=True, help="Registered Atom-JEPA models to finish")
parser.add_argument("--instance", default=None, help="Pin one instance type instead of the Atom-JEPA instance pools")
args = parser.parse_args()

for name in args.models:
    model = Model(name)
    existing = [e for e in model.endpoints() if Endpoint(e).exists()]

    if existing:
        end = Endpoint(existing[0])
        print(f"{name}: endpoint {existing[0]} already up, running inference only")
    else:
        print(f"Deploying {name}...")
        end = model.to_endpoint(tags=model.get_tags(), async_endpoint=True, instance=args.instance)
        end.set_owner("open_admet_cyp")

    end.test_inference()
    end.cross_fold_inference()
    print(f"{name}: cross-fold inference complete")
