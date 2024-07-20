# SageWorks Docker Image for Lambdas
Using the SageWorks Docker Image for AWS Lambda Jobs allows your Lambda Jobs to use and create AWS ML Pipeline Artifacts with SageWorks.

AWS, for some reason, does not allow Public ECRs to be used for Lambda Docker images. So you'll have to copy the Docker image into your private ECR. 

## Creating a Private ECR
You only need to do this if you don't already have a private ECR.

#### AWS Console to create Private ECR
1. Open the Amazon ECR console.
2. Choose "Create repository".
3. For "Repository name", enter `sageworks_base`.
4. Ensure "Private" is selected.
5. Choose "Create repository".

#### Command Line to create Private ECR
Create the ECR repository using the AWS CLI:

```bash
aws ecr create-repository --repository-name sageworks_base --region <region>
```

## Pulling Docker Image into Private ECR

**Note: You'll only need to do this when you want to update the SageWorks Docker image**

**Pull the SageWorks Public ECR Image**

```
docker pull public.ecr.aws/m6i5k1r2/sageworks_base:latest
```

**Tag the image for your private ECR**

```
docker tag public.ecr.aws/m6i5k1r2/sageworks_base:latest \
<your-account-id>.dkr.ecr.<region>.amazonaws.com/<private-repo>:latest
```

**Push the image to your private ECR**

```
aws ecr get-login-password --region <region> --profile <profile> | \
docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

docker push <account-id>.dkr.ecr.<region>.amazonaws.com/<private-repo>:<tag>
```

### Using the Docker Image for your Lambdas
Okay, now that you have the SageWorks Docker image in your private ECR, here's how you use that image for your Lambda jobs.

#### AWS Console
1. Open the AWS Lambda console.
2. Create a new function.
3. Select "Container image".
4. Use the ECR image URI: `<account-id>.dkr.ecr.<region>.amazonaws.com/<private-repo>:<tag>`.

#### Command Line
Create the Lambda function using the AWS CLI:

```bash
aws lambda create-function \
 --region <region> \
 --function-name myLambdaFunction \
 --package-type Image \
 --code ImageUri=<account-id>.dkr.ecr.<region>.amazonaws.com/<private-repo>:<tag> \
 --role arn:aws:iam::<account-id>:role/<execution-role>
```

#### Python CDK
Define the Lambda function in your CDK app:

```python
from aws_cdk import (
   aws_lambda as _lambda,
   core
)

class MyLambdaStack(core.Stack):
   def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
       super().__init__(scope, id, **kwargs)

       _lambda.Function(self, "MyLambdaFunction",
                        code=_lambda.Code.from_ecr_image("<account-id>.dkr.ecr.<region>.amazonaws.com/<private-repo>:<tag>"),
                        handler=_lambda.Handler.FROM_IMAGE,
                        runtime=_lambda.Runtime.FROM_IMAGE,
                        role=iam.Role.from_role_arn(self, "LambdaRole", "arn:aws:iam::<account-id>:role/<execution-role>"))

app = core.App()
MyLambdaStack(app, "MyLambdaStack")
app.synth()
```

#### Cloudformation
Define the Lambda function in your CloudFormation template.

```yaml
Resources:
 MyLambdaFunction:
   Type: AWS::Lambda::Function
   Properties:
     Code:
       ImageUri: <account-id>.dkr.ecr.<region>.amazonaws.com/<private-repo>:<tag>
     Role: arn:aws:iam::<account-id>:role/<execution-role>
     PackageType: Image
```