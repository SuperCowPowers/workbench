!!! tip inline end "Visibility and Control"
    The Workbench REPL provides AWS ML Pipeline visibility just like the [Workbench Dashboard](../aws_setup/dashboard_stack.md) but also provides control over the creation, modification, and deletion of artifacts through the Python API.

The Workbench REPL is a customized iPython shell. It provides tailored functionality for easy interaction with Workbench objects and since it's based on iPython developers will feel right at home using autocomplete, history, help, etc. Both easy and powerful, the Workbench REPL puts control of AWS ML Pipelines at your fingertips.

### Installation
`pip install workbench`

### Usage
Just type `workbench` at the command line and the Workbench shell will spin up and provide a command view of your AWS Machine Learning Pipelines.

At startup the Workbench shell, will connect to your AWS Account and create a summary of the Machine Learning artifacts currently residing on the account.

<img alt="workbench_repl" style="float: right; width: 450px; padding-left: 5px;"
src="https://github.com/SuperCowPowers/workbench/assets/4806709/10a969ed-3415-4d9f-ad0d-ac23706e6202">

**Available Commands:**

- status
- config
- incoming_data
- glue_jobs
- data_sources
- feature_sets
- models
- endpoints
- and more...


All of the [API Classes](../api_classes/overview.md) are auto-loaded, so drilling down on an individual artifact is easy. The same Python API is provided so if you want additional info on a **model**, for instance, simply create a model object and use any of the documented API methods.

```
m = Model("abalone-regression")
m.details()
<shows info about the model>
```


## Additional Resources

- Setting up Workbench on your AWS Account: [AWS Setup](../aws_setup/core_stack.md)
- Using Workbench for ML Pipelines: [Workbench API Classes](../api_classes/overview.md)

<img align="right" src="../images/scp.png" width="180">

- Workbench Core Classes: [Core Classes](../core_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
