!!! tip inline end "Workbench Lambda Layers"
    AWS Lambda Jobs are a great way to spin up data processing jobs. Follow this guide and empower AWS Lambda with Workbench!

Workbench makes creating, testing, and debugging of AWS Lambda Functions easy. The exact same [Workbench API Classes](../api_classes/overview.md) are used in your AWS Lambda Functions. Also since Workbench manages the access policies you'll be able to test new Lambda Jobs locally and minimizes surprises when deploying.
    
!!! warning inline end "Work In Progress"
    The Workbench Lambda Layers are a great way to use Workbench but they are still in 'beta' mode so please let us know if you have any issues.
    
## Lambda Job Setup

Setting up a AWS Lambda Job that uses Workbench is straight forward. Workbench can be 'installed' using a Lambda Layer and then you can use the Workbench API just like normal.

Here are the ARNs for the current Workbench Lambda Layers, please note they are specified with region and Python version in the name, so if your lambda is us-east-1, python 3.12, pick this ARN with those values in it.

### Python 3.12
 
**us-east-1**

- arn:aws:lambda:us-east-1:507740646243:layer:workbench_lambda_layer-us-east-1-python312:2

**us-west-2**

- arn:aws:lambda:us-west-2:507740646243:layer:workbench\_lambda\_layer-us-west-2-python312:3

**Note:** If you're using lambdas on a different region or with a different Python version, just let us know and we'll publish some additional layers.

<img alt="lambda_layer"  padding: 20px; border: 1px solid grey;""
src="https://github.com/user-attachments/assets/7d0e2fbe-b907-42bc-96bd-3b274d94c3de">

At the bottom of the Lambda page there's an 'Add Layer' button. You can click that button and specify the layer using the ARN above. Also in the 'General Configuration' set these parameters:

- Timeout: 5 Minutes
- Memory: 4096
- Ephemeral storage: 2048

**Set the WORKBENCH_BUCKET ENV**
Workbench will need to know what bucket to work out of, so go into the Configuration...Environment Variables... and add one for the Workbench bucket that your are using for AWS Account (dev, prod, etc).
<img alt="lambda_layer"  padding: 20px; border: 1px solid grey;""
src="https://github.com/user-attachments/assets/a5afdaff-188f-45ca-bd66-1ab62d7b0b2a">


!!! tip "Lambda Role Details"
    If your Lambda Function already use an existing IAM Role then you can add the Workbench policies to that Role to enable the Lambda Job to perform Workbench API Tasks. See [Workbench Access Controls](https://docs.google.com/presentation/d/1_KwbaBsyBoiWW_8SEallHg8RMsi9FdK10dr2wwzo3CA/edit?usp=sharing)

## Workbench Lambda Example
Here's a simple example of using Workbench in your Lambda Function. 

!!! warning "Workbench Layer is Compressed"
    The Workbench Lambda Layer is compressed (*to fit all the awesome*). This means that the `load_lambda_layer()` method must be called before using any other Workbench imports, see the example below. If you do not do this you'll probably get a `No module named 'numpy'` error or something like that.

```py title="examples/lambda_hello_world.py"
import json
from pprint import pprint
from workbench.utils.lambda_utils import load_lambda_layer
    
# Load/Decompress the Workbench Lambda Layer
load_lambda_layer()

# After 'load_lambda_layer()' we can use other Workbench imports
from workbench.api import Meta
from workbench.api import Model 

def lambda_handler(event, context):
    
    # Create our Meta Class and get a list of our Models
    meta = Meta()
    models = meta.models()
    
    print(f"Number of Models: {len(models)}")
    print(models)
        
    # Onboard a model
    model = Model("abalone-regression")
    pprint(model.details())
        
    # Return success
    return {
        'statusCode': 200,
        'body': { "incoming_event": event}
    }
```

## Exception Log Forwarding
When a Lambda Job crashes (has an exception), the AWS console will show you the last line of the exception, this is mostly useless. If you use Workbench log forwarding the exception/stack will be forwarded to CloudWatch.

```py
from workbench.utils.workbench_logging import exception_log_forward

with exception_log_forward():
   <my lambda code>
   ...
   <exception happens>
   <more of my code>
```
The `exception_log_forward` sets up a **context manager** that will trap exceptions and forward the exception/stack to CloudWatch for diagnosis. 

## Lambda Function Local Testing
Lambda Power without the Pain. Workbench manages the AWS Execution Role/Policies, so local API and Lambda Functions will have the same permissions/access. Also using the same Code as your notebooks or scripts makes creating and testing Lambda Functions a breeze.

```shell
python my_lambda_function.py --workbench-bucket <your bucket>
```

## Additional Resources
- Workbench Access Management: [Workbench Access Management](https://docs.google.com/presentation/d/1_KwbaBsyBoiWW_8SEallHg8RMsi9FdK10dr2wwzo3CA/edit?usp=sharing)
- Setting up Workbench on your AWS Account: [AWS Setup](../aws_setup/core_stack.md)

<img align="right" src="../images/scp.png" width="180">

- Using Workbench for ML Pipelines: [Workbench API Classes](../api_classes/overview.md)

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
