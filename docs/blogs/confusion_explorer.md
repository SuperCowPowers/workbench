# Confusion Explorer: Beyond the Confusion Matrix
!!! tip inline end "How Does Confidence Work?"
    The confidence slider is powered by VGMU — see our [Model Confidence](model_confidence.md) blog for the full details on how Workbench scores prediction uncertainty.

Classification models get a lot of mileage out of the standard confusion matrix — it's simple, familiar, and gives you a quick read on where your model is getting things right and wrong. But when you want to understand *why* certain compounds are being misclassified and *how confident* the model was when it got them wrong, the traditional matrix falls short. The Confusion Explorer pairs an enhanced confusion matrix with an interactive ternary probability plot, giving you a linked view that connects aggregate performance to individual predictions.

<figure style="margin: 20px auto; text-align: center;">
<img src="../../images/confusion_explorer/overview.png" alt="Confusion Explorer overview showing linked confusion matrix and ternary plot" style="max-width: 800px; width: 100%;">
<figcaption style="font-size: 0.85em;"><em>Residual-colored confusion matrix linked to a ternary probability plot. Cell opacity reflects count, color indicates error severity. Each point in the triangle is one prediction positioned by its class probabilities.</em></figcaption>
</figure>

## The Ternary Probability Plot

The right side of the explorer shows a **ternary plot** — each point is a single prediction, positioned according to its three class probabilities. Points in the corners have high confidence for a single class; points near the center are uncertain. The dashed lines partition the triangle into the three decision regions.

Coloring by **residual** (the default) makes misclassifications immediately visible. Correct predictions appear in blue/purple and cluster within their class's decision region. Green and red points are off-by-one and off-by-two errors, respectively. Even at a glance, you can see the spatial structure of your model's mistakes — misclassified compounds tend to sit near decision boundaries, exactly where you'd expect the model to struggle.

## Confidence-Aware Analysis

!!! tip inline end "Confidence Slider"
    The confidence slider filters both the matrix and triangle simultaneously. Drag the lower bound up to focus on only the predictions the model is most sure about.

A key insight in model evaluation is that **not all predictions deserve equal scrutiny**. A model that's uncertain about a compound and gets it wrong is behaving reasonably — that's an expected failure mode. But a model that's highly confident *and* wrong? That's where you should focus your attention.

The confidence slider at the top lets you filter predictions by their [confidence score](model_confidence.md). When you slide the lower bound up to 0.5 or higher, you're asking: "Among the predictions the model is most confident about, how does it perform?"

<figure style="margin: 20px auto; text-align: center;">
<img src="../../images/confusion_explorer/high_confidence.png" alt="High confidence filtering shows near-perfect classification" style="max-width: 800px; width: 100%;">
<figcaption style="font-size: 0.85em;"><em>Filtering to high-confidence predictions (above 0.5) yields zero misclassifications. Well-separated clusters in each corner confirm the model is accurate and confident.</em></figcaption>
</figure>

In the screenshot above, filtering to high-confidence predictions yields a perfect confusion matrix — zero misclassifications. The ternary plot confirms this: every point sits deep in its correct corner, far from the decision boundaries. This is exactly what you want to see — it means the model's confidence scores are well-calibrated and can be used to prioritize which predictions to trust.

## Drilling Down on Errors

Clicking any cell in the confusion matrix **filters the triangle** to show only the compounds in that cell. Non-matching compounds are dimmed, and the selected cell gets a highlight border. This is where the real investigative work happens.

<figure style="margin: 20px auto; text-align: center;">
<img src="../../images/confusion_explorer/drilldown_on_errors.png" alt="Drilling down on misclassified compounds with molecule hover" style="max-width: 800px; width: 100%;">
<figcaption style="font-size: 0.85em;"><em>Clicking the "low actual / high predicted" cell highlights those 5 errors on the triangle. Hovering reveals compound A-5550, a sulfonic acid misclassified from low to high solubility.</em></figcaption>
</figure>

In this example, we've clicked the off-diagonal cell showing 5 compounds that were actually "low" solubility but predicted as "high." The triangle plot zooms in on just those 5 points, and hovering reveals the molecular structure. This closes the loop from aggregate metrics to actionable chemistry — you can inspect exactly which compounds are confusing the model and look for structural patterns that might explain the misclassification.

## Visual Design Choices

A few design details that make the explorer more informative:

- **Residual coloring**: The confusion matrix uses the same colorscale as the ternary plot — diagonal cells (correct) are muted, off-diagonal cells get progressively warmer colors based on distance from the diagonal. This makes error severity visible at a glance.
- **Log-scaled opacity**: Cell opacity is proportional to `log(count)`, so high-count cells stand out while low-count cells (including zeros) fade into the background. This prevents rare misclassifications from visually competing with dominant cells.
- **Linked interaction**: The confidence slider and matrix clicks both act as filters — they update the triangle without resetting your color selection. You can switch between coloring by residual, prediction, confidence, or confusion without losing your current filter state.

## Under the Hood: VGMU Confidence

The confidence scores driving the slider come from **VGMU** (Variance-Gated Margin Uncertainty), which combines two signals from the 5-model ensemble: the probability margin between the top two classes and the ensemble's disagreement on those probabilities. This produces scores where high confidence means both a clear winner *and* model agreement — not just a high max probability. The scores are then calibrated via isotonic regression so that a confidence of 0.85 genuinely reflects ~85% accuracy. For the full details on how Workbench computes confidence for both classification and regression models, see our [Model Confidence](model_confidence.md#vgmu-variance-gated-margin-uncertainty) blog.

## References

- [Plotly Ternary Plots](https://plotly.com/python/ternary-plots/) — Plotly's ternary plot documentation, the visualization framework underlying the Confusion Explorer
- [Weiss et al., "Variance-Gated Ensembles: An Epistemic-Aware Framework" (2025)](https://arxiv.org/abs/2602.08142) — The VGMU approach for combining margin and ensemble variance in classification confidence
- [Galil et al., "What Can We Learn From The Selective Prediction And Uncertainty Estimation Performance Of 523 Imagenet Classifiers?" (2022)](https://arxiv.org/abs/2210.14070) — Analysis showing max probability alone is suboptimal for detecting incorrect predictions
- [scikit-learn ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) — The standard confusion matrix visualization that the Confusion Explorer builds upon

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
