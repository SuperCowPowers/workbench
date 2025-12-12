import pandas as pd
from workbench.api import Model

# List of all available models
model_list = [
    "caco-2-efflux-reg-xgb",
    "caco-2-efflux-reg-pytorch",
    "caco-2-efflux-reg-chemprop",
    "caco-2-efflux-reg-chemprop-hybrid",
    "caco-2-papp-a-b-reg-xgb",
    "caco-2-papp-a-b-reg-pytorch",
    "caco-2-papp-a-b-reg-chemprop",
    "caco-2-papp-a-b-reg-chemprop-hybrid",
    "hlm-clint-reg-xgb",
    "hlm-clint-reg-pytorch",
    "hlm-clint-reg-chemprop",
    "hlm-clint-reg-chemprop-hybrid",
    "ksol-reg-xgb",
    "ksol-reg-pytorch",
    "ksol-reg-chemprop",
    "ksol-reg-chemprop-hybrid",
    "logd-reg-xgb",
    "logd-reg-pytorch",
    "logd-reg-chemprop",
    "logd-reg-chemprop-hybrid",
    "mbpb-reg-xgb",
    "mbpb-reg-pytorch",
    "mbpb-reg-chemprop",
    "mbpb-reg-chemprop-hybrid",
    "mgmb-reg-xgb",
    "mgmb-reg-pytorch",
    "mgmb-reg-chemprop",
    "mgmb-reg-chemprop-hybrid",
    "mlm-clint-reg-xgb",
    "mlm-clint-reg-pytorch",
    "mlm-clint-reg-chemprop",
    "mlm-clint-reg-chemprop-hybrid",
    "mppb-reg-xgb",
    "mppb-reg-pytorch",
    "mppb-reg-chemprop",
    "mppb-reg-chemprop-hybrid",
]

# First pull/store metrics from the multi-task model
runs = [
    "cv_caco_2_efflux",
    "cv_caco_2_papp_a_b",
    "cv_hlm_clint",
    "cv_ksol",
    "cv_logd",
    "cv_mbpb",
    "cv_mgmb",
    "cv_mlm_clint",
    "cv_mppb",
]
model = Model("open-admet-chemprop-mt")
mt_metrics = {}
for run in runs:
    metrics = model.get_inference_metrics(run).reset_index()
    print(f"Metrics for run {run}:")
    print(metrics)
    model_name = run.replace("cv_", "").replace("_", "-") + "-mt"
    mt_metrics[model_name] = metrics
    print(model_name)
    print(metrics)

# For each model, load it and get its full cross-fold inference metrics
metric_rows = []
for model_name in model_list:
    model = Model(model_name)
    metrics = model.get_inference_metrics("full_cross_fold").reset_index()
    print(metrics)

    # Now convert to DataFrame rows (with model name + metrics)
    row = {"model_name": model_name}
    row.update(metrics.iloc[0].to_dict())
    metric_rows.append(row)


# Create a DataFrame of all metrics
metrics_df = pd.DataFrame(metric_rows)
print(metrics_df)

# Save to CSV
metrics_df.round(2).to_csv("model_cross_fold_metrics.csv", index=False)
print("Saved model cross-fold metrics to model_cross_fold_metrics.csv")
