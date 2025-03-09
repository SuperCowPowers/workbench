import numpy as np
import seaborn as sns
import json

cmap = sns.light_palette("#97c3d5", as_cmap=True)  # Light Blue
# cmap = sns.light_palette("#175676", as_cmap=True)  # Dark Blue
# cmap = sns.light_palette("#F46036",as_cmap=True) #orange
# cmap = sns.light_palette("#8dc9b7",as_cmap=True) #green
vals = np.linspace(0, 1, 8)
plotly_scale = []
for v in vals:
    r, g, b, a = cmap(v)
    plotly_scale.append([round(v, 2), f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"])

for sublist in plotly_scale:
    print(json.dumps(sublist) + ",")
