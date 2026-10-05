"""Delete the CYP models whose experiments are finished.

Every name here has its verdict recorded in `docs/planning/openadmet_cyp_challenge.md`; the
model is the evidence, the doc is the result, and only one of those needs keeping. Anything in
the standing ensemble, the seed replicates behind the resolution thresholds, and the readout
model are deliberately absent from this list.

Endpoints go before their models, since a model with a live endpoint cannot be removed.

    python cyp_cleanup.py            # list what would go, delete nothing
    python cyp_cleanup.py --apply
"""

import argparse

from workbench.api import Endpoint, Model

REMOVE = {
    "fingerprint XGB, lowers the CYP2D6 pool": [
        "cyp-fp-reg-xgb-1a2",
        "cyp-fp-reg-xgb-2c9",
        "cyp-fp-reg-xgb-2d6",
        "cyp-fp-reg-xgb-3a4",
    ],
    "2D+3D-v2 XGB, lowers the CYP2D6 pool": [
        "cyp-2d-3dv2-reg-xgb-1a2",
        "cyp-2d-3dv2-reg-xgb-2c9",
        "cyp-2d-3dv2-reg-xgb-2d6",
        "cyp-2d-3dv2-reg-xgb-3a4",
        "cyp-2d-3dv2-reg-xgb-2c9-261004",
        "cyp-2d-3dv2-reg-xgb-2c9-261005",
    ],
    "binary activity heads": ["cyp-reg-chemprop-union-p30-act"],
    "ChEMBL censored bounds, with the loss on and off": [
        "cyp-reg-chemprop-union-p30-cen",
        "cyp-reg-chemprop-union-p30-cenlabels",
    ],
    "public-weight A/B, 0.30 against 0.05": [
        "cyp-reg-chemprop-union-p30-h26",
        "cyp-reg-chemprop-union-p05-h26",
    ],
    "public potency pooled into the scored column": ["cyp-reg-chemprop-2d6-pooled"],
    "the TDI arm as its own target": ["cyp-reg-chemprop-2d6-tdi"],
    "CheMeleon pretrained encoder": [
        "cyp-reg-chemprop-2d6-single-chemeleon-fz0",
        "cyp-reg-chemprop-2d6-single-chemeleon-fz10",
    ],
    "rebuilds that came back lower on all four isoforms": [
        "cyp-reg-chemprop-mt-aux-100-r2",
        "cyp-reg-chemprop-2d6-isoform-r2",
        "cyp-reg-chemprop-union-p30-tox-h36",
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually delete; otherwise just list")
    args = parser.parse_args()

    for reason, names in REMOVE.items():
        print(f"\n{reason}")
        for name in names:
            model, end = Model(name), Endpoint(name)
            state = "model" + (" + endpoint" if end.exists() else "")
            if not model.exists():
                print(f"  {name:<46} already gone")
                continue
            if not args.apply:
                print(f"  {name:<46} would delete {state}")
                continue
            if end.exists():
                end.delete()
            model.delete()
            print(f"  {name:<46} deleted {state}")

    total = sum(len(v) for v in REMOVE.values())
    print(f"\n{total} models" + (" deleted" if args.apply else " would be deleted — re-run with --apply"))
