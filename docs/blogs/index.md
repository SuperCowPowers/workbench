# Workbench Blogs
!!! tip inline end "Just Getting Started?"
    Workbench Blogs are a great way to see what's possible with Workbench. When you're ready to jump in, the [API Classes](../api_classes/overview.md) will give you details on the Workbench ML Pipeline Classes.

Workbench blogs highlight interesting functionality and approaches that might be useful to a broader audience. Each blog gives a high-level overview of the topic with drilldowns into the trickier bits. Whether you're looking for implementation details, architecture decisions, or practical tips for deploying ML models on AWS, these posts cover the real-world challenges we've solved.

## Blogs

- **[Confusion Explorer: Beyond the Confusion Matrix](confusion_explorer.md):** The standard confusion matrix tells you *what* your model gets wrong — the Confusion Explorer shows you *why*. We pair a residual-colored matrix with an interactive ternary probability plot, linked through a confidence slider. Filter to high-confidence predictions, click a cell to isolate misclassified compounds, and hover to see molecular structures.

- **[Model Confidence: Building on Conformal Prediction](model_confidence.md):** How does Workbench approach prediction uncertainty? We walk through our current pipeline — 5-fold ensemble disagreement, conformal calibration for coverage guarantees, and percentile-rank confidence scoring — discuss the trade-offs, and point to the foundational work we're building on.

- **[SHAP Values for ChemProp Models](chemprop_shap.md):** How do you explain a graph neural network? In this blog we explore our per-bit ablation approach for computing [SHAP values](https://shap.readthedocs.io/) on [ChemProp](https://github.com/chemprop/chemprop) MPNN models. We walk through the technical approach, show key code snippets, and analyze real SHAP output from a LogD regression model — validating that the model independently learns known structure-lipophilicity relationships.

- **[Multi-Task ChemProp: Two Mechanisms, One Model](chemprop_multi_task.md):** Multi-task ChemProp lifts come from two distinct mechanisms — transfer learning through a shared MPNN encoder, and chemical-space expansion from non-overlapping auxiliary compounds. We separate the two with conceptual diagrams, show when each one helps (vs. hurts), and tie the discussion back to Workbench's pre-flight `assess_multi_task_data` utility that scores both axes from labels alone.

- **[Inside a Workbench AWS Endpoint](aws_endpoint_architecture.md):** A deep dive into endpoint architecture — comparing the default SageMaker stack (Nginx, Gunicorn, Flask) with Workbench's modern ASGI stack (Uvicorn, FastAPI). We cover the custom image with pre-loaded chemistry packages, and Workbench's binary-search error handling that isolates bad rows instead of failing entire batches.

- **[Molecular Standardization](molecular_standardization.md):** Why molecular standardization matters for ML pipelines. We walk through Workbench's four-step pipeline — cleanup, salt handling, charge neutralization, and tautomer canonicalization. We also describe **2D** and **3D** molecular descriptors computed by our feature endpoints.

- **[A Weekend on the OpenADMET PXR Challenge](pxr_weekend_experiments.md):** Five experiments against a plain from-scratch Chemprop D-MPNN on PXR induction. Four added complexity for its own sake — 2D-vs-3D descriptor ablations, a from-scratch xTB 3D-descriptor rebuild, a multi-model ensemble, a CheMeleon foundation-model warm-start — and all four lost on the revealed held-out analog series. The fifth, multi-task learning on logD/logP (PXR's lipophilicity driver), was the one that finally beat the baseline. The lesson: a strong simple baseline plus honest OOD evaluation is hard to beat, and complexity only pays off when it's aligned with the mechanism. Includes the SHAP "high-importance ≠ generalizes" paradox.

- **[3D Molecular Descriptors](3d_descriptors.md):** A deep dive into Workbench's 3D descriptor pipeline — from conformer generation (ETKDGv3 with tiered fallback and MMFF94s optimization) to 75 features covering molecular shape, charged partial surface area, pharmacophore spatial distribution, and conformational flexibility. We cover the production guardrails (complexity checks, per-molecule timeouts) that keep it reliable as a deployed endpoint.

- **[Feature Endpoints: From Training to LiveDesign](feature_endpoints.md):** How Workbench uses SageMaker-hosted feature endpoints to guarantee identical feature computation — whether the request comes from a training pipeline, an inference endpoint, or a drug discovery platform like LiveDesign or StarDrop. We compare this approach to feature stores, platform UDFs (Databricks/Tecton), and open-source alternatives.

## Questions?
<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
