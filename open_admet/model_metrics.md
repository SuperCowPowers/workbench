# Meta-Model Performance Summary
We used 5 different model types for each ADMET endpoint. The [Workbench AWS Dashboard](https://aws.amazon.com/marketplace/pp/prodview-5idedc7uptbqo) supports all 5 of these model types and makes the creation, training, and deployment of models into your production AWS accounts a snap.

- **XGBoost:** Gradient boosted trees on RDKit molecular descriptors
- **PyTorch:** Neural network on RDKit molecular descriptors
- **ChemProp:** Message Passing Neural Network (MPNN) on molecular graphs
- **ChemProp Hybrid:** MPNN + Top RDKit descriptors combined
- **ChemProp Multi-Task:** Single MPNN predicting all 9 endpoints simultaneously

## Individual Model Results

| Model                                 | MAE   | R²    | RMSE  | MedAE |
|---------------------------------------|-------|-------|-------|-------|
| Caco2-Efflux                          |       |       |       |       |
| - caco-2-efflux-reg-xgb               | 0.208 | 0.349 | 0.299 | 0.163 |
| - caco-2-efflux-reg-pytorch           | 0.201 | 0.386 | 0.291 | 0.156 |
| - caco-2-efflux-reg-chemprop          | 0.183 | 0.466 | 0.271 | 0.142 |
| - caco-2-efflux-reg-chemprop-hybrid   | 0.185 | 0.458 | 0.273 | 0.143 |
| - caco-2-efflux-mt                    | 0.128 | 0.658 | 0.197 | 0.077 |
| Caco2-Papp-A>B                        |       |       |       |       |
| - caco-2-papp-a-b-reg-xgb             | 0.263 | 0.512 | 0.381 | 0.196 |
| - caco-2-papp-a-b-reg-pytorch         | 0.257 | 0.531 | 0.373 | 0.193 |
| - caco-2-papp-a-b-reg-chemprop        | 0.232 | 0.608 | 0.341 | 0.174 |
| - caco-2-papp-a-b-reg-chemprop-hybrid | 0.224 | 0.625 | 0.334 | 0.167 |
| - caco-2-papp-a-b-mt                  | 0.187 | 0.661 | 0.248 | 0.145 |
| HLM CLint                             |       |       |       |       |
| - hlm-clint-reg-xgb                   | 0.345 | 0.540 | 0.500 | 0.253 |
| - hlm-clint-reg-pytorch               | 0.343 | 0.548 | 0.495 | 0.256 |
| - hlm-clint-reg-chemprop              | 0.320 | 0.605 | 0.463 | 0.240 |
| - hlm-clint-reg-chemprop-hybrid       | 0.321 | 0.602 | 0.465 | 0.241 |
| - hlm-clint-mt                        | 0.293 | 0.614 | 0.391 | 0.222 |
| KSOL                                  |       |       |       |       |
| - ksol-reg-xgb                        | 0.477 | 0.680 | 0.708 | 0.337 |
| - ksol-reg-pytorch                    | 0.494 | 0.654 | 0.736 | 0.352 |
| - ksol-reg-chemprop                   | 0.451 | 0.713 | 0.670 | 0.318 |
| - ksol-reg-chemprop-hybrid            | 0.439 | 0.727 | 0.654 | 0.308 |
| - ksol-chemprop-mt                    | 0.294 | 0.651 | 0.426 | 0.181 |
| LogD                                  |       |       |       |       |
| - logd-reg-xgb                        | 0.343 | 0.783 | 0.514 | 0.251 |
| - logd-reg-pytorch                    | 0.350 | 0.777 | 0.521 | 0.256 |
| - logd-reg-chemprop                   | 0.319 | 0.811 | 0.479 | 0.233 |
| - logd-reg-chemprop-hybrid            | 0.311 | 0.819 | 0.469 | 0.226 |
| - logd-chemprop-mt                    | 0.312 | 0.875 | 0.422 | 0.241 |
| MBPB                                  |       |       |       |       |
| - mbpb-reg-xgb                        | 0.243 | 0.463 | 0.344 | 0.182 |
| - mbpb-reg-pytorch                    | 0.238 | 0.479 | 0.339 | 0.178 |
| - mbpb-reg-chemprop                   | 0.218 | 0.555 | 0.313 | 0.163 |
| - mbpb-reg-chemprop-hybrid            | 0.213 | 0.570 | 0.308 | 0.159 |
| - mbpb-reg-chemprop-mt                | 0.154 | 0.790 | 0.202 | 0.120 |
| MGMB                                  |       |       |       |       |
| - mgmb-reg-xgb                        | 0.156 | 0.238 | 0.217 | 0.117 |
| - mgmb-reg-pytorch                    | 0.152 | 0.259 | 0.214 | 0.114 |
| - mgmb-reg-chemprop                   | 0.135 | 0.387 | 0.195 | 0.101 |
| - mgmb-reg-chemprop-hybrid            | 0.130 | 0.416 | 0.190 | 0.097 |
| - mgmb-reg-chemprop-mt                | 0.133 | 0.776 | 0.185 | 0.092 |
| MLM CLint                             |       |       |       |       |
| - mlm-clint-reg-xgb                   | 0.378 | 0.498 | 0.548 | 0.280 |
| - mlm-clint-reg-pytorch               | 0.371 | 0.514 | 0.539 | 0.274 |
| - mlm-clint-reg-chemprop              | 0.347 | 0.567 | 0.509 | 0.256 |
| - mlm-clint-reg-chemprop-hybrid       | 0.341 | 0.579 | 0.502 | 0.251 |
| - mlm-clint-mt                        | 0.333 | 0.683 | 0.451 | 0.252 |
| MPPB                                  |       |       |       |       |
| - mppb-reg-xgb                        | 0.205 | 0.619 | 0.290 | 0.153 |
| - mppb-reg-pytorch                    | 0.204 | 0.620 | 0.290 | 0.152 |
| - mppb-reg-chemprop                   | 0.183 | 0.682 | 0.265 | 0.136 |
| - mppb-reg-chemprop-hybrid            | 0.178 | 0.692 | 0.261 | 0.132 |
| - mbpb-reg-chemprop-mt                | 0.178 | 0.744 | 0.234 | 0.148 |


## Multi-Task ChemProp Model Results
We have a a few additional metrics for the ChemProp Multi-Task model, including Spearman correlation and support (N):

| Property        | MAE   | R²    | RMSE  | MedAE | Spearman | N    |
|-----------------|-------|-------|-------|-------|----------|------|
| Caco-2 Efflux   | 0.128 | 0.658 | 0.197 | 0.077 | 0.741    | 2161 |
| Caco-2 Papp A>B | 0.187 | 0.661 | 0.248 | 0.145 | 0.807    | 2157 |
| HLM CLint       | 0.293 | 0.614 | 0.391 | 0.222 | 0.801    | 3759 |
| KSOL            | 0.294 | 0.651 | 0.426 | 0.181 | 0.742    | 5128 |
| LogD            | 0.312 | 0.875 | 0.422 | 0.241 | 0.935    | 5039 |
| MBPB            | 0.154 | 0.790 | 0.202 | 0.120 | 0.898    | 975  |
| MGMB            | 0.133 | 0.776 | 0.185 | 0.092 | 0.893    | 222  |
| MLM CLint       | 0.333 | 0.683 | 0.451 | 0.252 | 0.822    | 4522 |
| MPPB            | 0.178 | 0.744 | 0.234 | 0.148 | 0.865    | 1302 |