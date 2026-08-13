"""Capture the held-out ExpansionRx test inference run on the OpenADMET champions.

The promotion script only runs test_inference() and cross_fold_inference() on the model
it promotes, so a dated champion copy (e.g. logd-reg-chemprop-260729) lands without the
rx_test_<target> capture that its challengers all carry. Without it the champion cannot
be compared to a challenger on held-out data -- the contest reports fall back to
full_cross_fold, which is out-of-fold training data, not the ExpansionRx test rows.

This script closes that gap. For each of the nine assay contests it resolves the model
currently served by <assay>-reg-v1 and scores it on the held-out test rows, persisting
the run as rx_test_<target>.

Re-running is safe: a capture of the same name is overwritten in place.

The test rows come pre-featurized and pre-log-transformed from the DFStore key written by
../../load_data/load_data.py -- the same source the training scripts score against, so
champion and challenger numbers are directly comparable. Do NOT re-derive the features or
the target transform here; drift between two copies of that logic is exactly the failure
this shared key prevents. (For the record, the transforms are log10(uM + 1) - 6 for ksol
and caco2_papp_a_b, log10(x + 1) for the other seven assays, and identity for logd -- see
workbench.utils.chem_utils.misc.micromolar_to_log.)

The featurized columns cover every framework in the family: chemprop champions consume
smiles alone, while chemprop-desc, pytorch and xgb champions also need their RDKit/
Mordred descriptor columns. Each champion's own feature_list decides what gets sent.
"""

from workbench.api import DFStore, Endpoint, Model

# Featurized + log-transformed ExpansionRx test set, written by ../../load_data/load_data.py
TEST_STORE_KEY = "/workbench/datasets/open_admet_rx_test_featurized"

# assay target -> the endpoint whose served model is the champion
CHAMPION_ENDPOINTS = {
    "caco2_efflux": "caco2-efflux-reg-v1",
    "caco2_papp_a_b": "caco2-papp-a-b-reg-v1",
    "hlm_clint": "hlm-clint-reg-v1",
    "ksol": "ksol-reg-v1",
    "logd": "logd-reg-v1",
    "mbpb": "mbpb-reg-v1",
    "mgmb": "mgmb-reg-v1",
    "mlm_clint": "mlm-clint-reg-v1",
    "mppb": "mppb-reg-v1",
}


def capture_test_inference(end: Endpoint, target: str, features: list) -> None:
    """Score the held-out ExpansionRx test rows for this champion and persist the run."""
    test_df = DFStore().get(TEST_STORE_KEY).dropna(subset=[target])
    columns = ["molecule_name", "smiles", target] + features
    print(f"Running held-out test inference for {end.name} on {len(test_df)} rows")
    end.inference(test_df[columns], capture_name=f"rx_test_{target}")


def main():
    for target, endpoint_name in CHAMPION_ENDPOINTS.items():
        end = Endpoint(endpoint_name)

        # The endpoint's input model IS the champion (a dated copy made at promotion time)
        champion = end.get_input()
        model = Model(champion)

        # smiles is passed explicitly by the helper; send the descriptor features only
        features = [f for f in model.features() if f != "smiles"]
        print(f"\n{target}: champion {champion} ({len(features)} descriptor features)")

        capture_test_inference(end, target, features)

        metrics = model.get_inference_metrics(f"rx_test_{target}")
        print(metrics)


if __name__ == "__main__":
    main()
