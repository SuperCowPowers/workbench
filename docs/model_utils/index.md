# Model Utilities
!!! tip inline end "Examples"
    Examples of using the Model Utilities are listed at the bottom of this page [Examples](#examples).

    
::: workbench.utils.model_utils


## Examples

### Feature Importance

```py 
"""Example for using some Model Utilities"""
from workbench.utils.model_utils import feature_importance

model = Model("aqsol_classification")
feature_importance(model)

```

**Output**

```py
[('mollogp', 469.0),
 ('minabsestateindex', 277.0),
 ('peoe_vsa8', 237.0),
 ('qed', 237.0),
 ('fpdensitymorgan1', 230.0),
 ('fpdensitymorgan3', 221.0),
 ('estate_vsa4', 220.0),
 ('bcut2d_logphi', 218.0),
 ('vsa_estate5', 218.0),
 ('vsa_estate4', 209.0),
```

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

- Workbench API Classes: [API Classes](../api_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
