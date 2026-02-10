# Workbench Blogs
!!! tip inline end "Just Getting Started?"
    Workbench Blogs are a great way to see what's possible with Workbench. When you're ready to jump in, the [API Classes](../api_classes/overview.md) will give you details on the Workbench ML Pipeline Classes.

Workbench blogs highlight interesting functionality and approaches that might be useful to a broader audience. Each blog gives a high-level overview of the topic with drilldowns into the trickier bits. Whether you're looking for implementation details, architecture decisions, or practical tips for deploying ML models on AWS, these posts cover the real-world challenges we've solved.

## Blogs

- **[Model Confidence: Building on Conformal Prediction](model_confidence.md):** How does Workbench approach prediction uncertainty? We walk through our current pipeline — 5-fold ensemble disagreement, conformal calibration for coverage guarantees, and percentile-rank confidence scoring — discuss the trade-offs, and point to the foundational work we're building on.

- **[SHAP Values for ChemProp Models](chemprop_shap.md):** How do you explain a graph neural network? In this blog we explore our per-bit ablation approach for computing [SHAP values](https://shap.readthedocs.io/) on [ChemProp](https://github.com/chemprop/chemprop) MPNN models. We walk through the technical approach, show key code snippets, and analyze real SHAP output from a LogD regression model — validating that the model independently learns known structure-lipophilicity relationships.

- **[Inside a Workbench AWS Endpoint](aws_endpoint_architecture.md):** A deep dive into endpoint architecture — comparing the default SageMaker stack (Nginx, Gunicorn, Flask) with Workbench's modern ASGI stack (Uvicorn, FastAPI). We cover the custom image with pre-loaded chemistry packages, and Workbench's binary-search error handling that isolates bad rows instead of failing entire batches.

- **[Canonicalization and Tautomerization](molecular_standardization.md):** A deep dive into why molecular standardization matters for ML pipelines. We look at the AqSol solubility dataset, compute molecular descriptors with RDKit, and explore why NaNs and parse errors show up on about 9% of compounds.

## Questions?
<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
