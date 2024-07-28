# SageWorks Sandbox AWS Deployment

To run/deploy the SageWorks AWS Sandbox you'll need install a couple of Python packages
either in a Python VirtualENV of your choice (PyENV is good) or any Python3 will do. You'll also need to use a newer version of node such as node v19.6

```
pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template and deploy the SageWorks Sandbox AWS Components.

```
cdk synth
cdk diff
cdk deploy
```

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

## CDK Notes
The `cdk.json` file tells the CDK Toolkit how to execute your app.

To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

