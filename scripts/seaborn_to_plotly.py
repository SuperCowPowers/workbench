import numpy as np
import seaborn as sns
import json

cmap = sns.light_palette("#97c3d5", as_cmap=True)
vals = np.linspace(0, 1, 8)
plotly_scale = []
for v in vals:
    r, g, b, a = cmap(v)
    plotly_scale.append([round(v, 2), f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"])

for sublist in plotly_scale:
    print(json.dumps(sublist))
