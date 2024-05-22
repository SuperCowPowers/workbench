# Residual Analysis

Important Reasons for High Residuals, most of these are common across domains and different modeling use cases. The **Activity Cliff** is specific to chemical compounds and drug discovery.

**Sparse Data Regions**

The observation is in a 'chemical space' with little or no nearby training observations, leading to poor generalization in these regions and resulting in high prediction errors.


**Noisy or Inconsistent Data**

The observation is in a chemical space where the training data is noisy, incorrect, or has high variance in the target variable, leading to unreliable predictions and high residuals.

**Activity Cliffs**

Structurally similar compounds exhibiting significantly different activities, making accurate prediction challenging due to steep changes in activity with minor structural modifications.

**Feature Engineering Issues**

Irrelevant or redundant features and poor feature scaling can negatively impact the model's performance and accuracy, resulting in higher residuals.

**Model Over/Under Fit**

Overfitting occurs when the model is too complex and captures noise, while underfitting happens when the model is too simple and misses underlying patterns, both leading to inaccurate predictions.