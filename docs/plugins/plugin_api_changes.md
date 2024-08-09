# Recent API Changes
Since we've recently introduced a **View()** class for DataSources and FeatureSets we needed to rename a few classes/modules.

### FeatureSets
For setting holdout ids we've changed that method name from `create_training_view()` to `set_training_holdouts()`

### Web/Plugins
We've changed the Web/UI View class to 'WebView'. So anywhere where you used to have **view** just replace with **web_view**

```
from sageworks.views.artifacts_view import ArtifactsView
```
is now

```
from sageworks.web_views.artifacts_web_view import ArtifactsWebView
```