# Recent API Changes
Since we've recently introduced a **View()** class for DataSources and FeatureSets we needed to rename a few classes/modules.

### FeatureSets
For setting holdout ids we've changed/combined to just one method `set_training_holdouts()`, so if you're using `create_training_view()` or `set_holdout_ids()` you can now just use the unified method `set_training_holdouts()`.

There's also a change to getting the training view table method.

```
old: fs.get_training_view_table(create=False)
new: fs.get_training_view_table(), does not need the create=False
```

### Models
```
inference_predictions() --> get_inference_predictions()
```

### Web/Plugins
We've changed the Web/UI View class to 'WebView'. So anywhere where you used to have **view** just replace with **web_view**

```
from sageworks.views.artifacts_view import ArtifactsView
```
is now

```
from sageworks.web_views.artifacts_web_view import ArtifactsWebView
```