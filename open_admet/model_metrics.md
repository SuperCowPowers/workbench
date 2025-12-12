# Meta-Model Performance Summary
We used 5 different model types for each ADMET endpoint. The [Workbench AWS Dashboard](https://aws.amazon.com/marketplace/pp/prodview-5idedc7uptbqo) supports all 5 of these model types and makes the creation, training, and deployment of models into your production AWS accounts a snap.

- **XGBoost:** Gradient boosted trees on RDKit molecular descriptors
- **PyTorch:** Neural network on RDKit molecular descriptors
- **ChemProp:** Message Passing Neural Network (MPNN) on molecular graphs
- **ChemProp Hybrid:** MPNN + Top RDKit descriptors combined
- **ChemProp Multi-Task:** Single MPNN predicting all 9 endpoints simultaneously

## Individual Model Results

| Model                               | MAE   | RMSE  | MedAE | R²    | N    |
|-------------------------------------|-------|-------|-------|-------|------|
| Caco2-Efflux                        |           
| caco-2-efflux-reg-xgb               | 0.15  | 0.23  | 0.09  | 0.52  | 2161 |
| caco-2-efflux-reg-pytorch           | 0.14  | 0.21  | 0.08  | 0.63  | 2161 |
| caco-2-efflux-reg-chemprop          | 0.13  | 0.20  | 0.08  | 0.66  | 2161 |
| caco-2-efflux-reg-chemprop-hybrid   | 0.13  | 0.19  | 0.08  | 0.67  | 2161 |
| caco-2-efflux-reg-mt                | 0.128 | 0.197 | 0.077 | 0.658 | 2161 |
| Caco2-Papp-A>B                      |
| caco-2-papp-a-b-reg-xgb             | 0.22  | 0.28  | 0.17  | 0.57  | 2157 |
| caco-2-papp-a-b-reg-pytorch         | 0.24  | 0.30  | 0.19  | 0.49  | 2157 |
| caco-2-papp-a-b-reg-chemprop        | 0.18  | 0.24  | 0.14  | 0.68  | 2157 |
| caco-2-papp-a-b-reg-chemprop-hybrid | 0.18  | 0.24  | 0.14  | 0.67  | 2157 |
| caco-2-papp-a-b-reg-mt              | 0.187 | 0.248 | 0.145 | 0.661 | 2157 |
| HLM CLint                           |
| hlm-clint-reg-xgb                   | 0.32  | 0.43  | 0.23  | 0.53  | 3759 |
| hlm-clint-reg-pytorch               | 0.31  | 0.41  | 0.24  | 0.58  | 3759 |
| hlm-clint-reg-chemprop              | 0.28  | 0.37  | 0.21  | 0.65  | 3759 |
| hlm-clint-reg-chemprop-hybrid       | 0.28  | 0.38  | 0.21  | 0.64  | 3759 |
| hlm-clint-reg-mt                    | 0.293 | 0.391 | 0.222 | 0.614 | 3759 |
| KSOL                                |
| ksol-reg-xgb                        | 0.31  | 0.46  | 0.18  | 0.60  | 5128 |
| ksol-reg-pytorch                    | 0.31  | 0.44  | 0.20  | 0.64  | 5128 |
| ksol-reg-chemprop                   | 0.27  | 0.40  | 0.16  | 0.70  | 5128 |
| ksol-reg-chemprop-hybrid            | 0.27  | 0.40  | 0.16  | 0.69  | 5128 |
| ksol-reg-mt                         | 0.294 | 0.426 | 0.181 | 0.651 | 5128 |
| LogD                                |
| logd-reg-xgb                        | 0.37  | 0.52  | 0.28  | 0.81  | 5039 |
| logd-reg-pytorch                    | 0.30  | 0.41  | 0.23  | 0.88  | 5039 |
| logd-reg-chemprop                   | 0.22  | 0.31  | 0.16  | 0.93  | 5039 |
| logd-reg-chemprop-hybrid            | 0.23  | 0.32  | 0.16  | 0.93  | 5039 |
| logd-reg-mt                         | 0.312 | 0.422 | 0.241 | 0.875 | 5039 |
| MBPB                                |
| mbpb-reg-xgb                        | 0.17  | 0.22  | 0.12  | 0.74  | 975  |
| mbpb-reg-pytorch                    | 0.15  | 0.20  | 0.12  | 0.79  | 975  |
| mbpb-reg-chemprop                   | 0.13  | 0.17  | 0.09  | 0.84  | 975  |
| mbpb-reg-chemprop-hybrid            | 0.13  | 0.18  | 0.09  | 0.84  | 975  |
| mbpb-reg-mt                         | 0.154 | 0.202 | 0.12  | 0.79  | 975  |
| MGMB                                |
| mgmb-reg-xgb                        | 0.16  | 0.22  | 0.11  | 0.68  | 222  |
| mgmb-reg-pytorch                    | 0.18  | 0.23  | 0.13  | 0.64  | 222  |
| mgmb-reg-chemprop                   | 0.14  | 0.19  | 0.11  | 0.75  | 222  |
| mgmb-reg-chemprop-hybrid            | 0.14  | 0.19  | 0.10  | 0.76  | 222  |
| mgmb-reg-mt                         | 0.133 | 0.185 | 0.092 | 0.776 | 222  |
| MLM CLint                           |
| mlm-clint-reg-xgb                   | 0.35  | 0.48  | 0.25  | 0.63  | 4522 |
| mlm-clint-reg-pytorch               | 0.34  | 0.46  | 0.26  | 0.67  | 4522 |
| mlm-clint-reg-chemprop              | 0.31  | 0.43  | 0.22  | 0.72  | 4522 |
| mlm-clint-reg-chemprop-hybrid       | 0.32  | 0.44  | 0.24  | 0.70  | 4522 |
| mlm-clint-reg-mt                    | 0.333 | 0.451 | 0.252 | 0.683 | 4522 |
| MPPB                                |
| mppb-reg-xgb                        | 0.19  | 0.25  | 0.15  | 0.70  | 1302 |
| mppb-reg-pytorch                    | 0.18  | 0.24  | 0.14  | 0.72  | 1302 |
| mppb-reg-chemprop                   | 0.17  | 0.23  | 0.13  | 0.75  | 1302 |
| mppb-reg-chemprop-hybrid            | 0.17  | 0.23  | 0.13  | 0.76  | 1302 |
| mppb-reg-mt                         | 0.178 | 0.234 | 0.148 | 0.744 | 1302 |
