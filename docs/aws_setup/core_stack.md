# Workbench AWS Stack
Welcome to the Workbench AWS Setup Guide. Workbench is deployed as an AWS **Stack** following the well architected system practices of AWS. 

!!! warning "AWS Setup can be a bit complex"
    Setting up Workbench with AWS can be a bit complex, but this only needs to be done ONCE for your entire company. The install uses standard CDK --> AWS Stacks and Workbench tries to make it straight forward. If you have any troubles at all feel free to contact us a [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on [Discord](https://discord.gg/WHAJuz8sw8) and we're happy to help you with AWS for FREE.
    
## Two main options when using Workbench
1. Spin up a new AWS Account for the Workbench Stacks ([Make a New Account](aws_tips_and_tricks.md))
2. Deploy Workbench Stacks into your existing AWS Account

Either of these options are fully supported, but we highly suggest a NEW account as it gives the following benefits:

- **AWS Data Isolation:** Data Scientists will feel empowered to play in the sandbox without impacting production services.
- **AWS Cost Accounting:** Monitor and Track all those new ML Pipelines that your team creates with Workbench :)

## Setting up Users and Groups
If your AWS Account already has users and groups set up you can skip this but here's our recommendations on setting up [SSO Users and Groups](aws_tips_and_tricks.md)

## Onboarding Workbench to your AWS Account

Pulling down the Workbench Repo
  ```
  git clone https://github.com/SuperCowPowers/workbench.git
  ```

## Workbench uses AWS Python CDK for Deployments
If you don't have AWS CDK already installed you can do these steps:

Mac

  ```
  brew install node 
  npm install -g aws-cdk
  ```
Linux

  ```
  sudo apt install nodejs
  sudo npm install -g aws-cdk
  ```
For more information on Linux installs see [Digital Ocean NodeJS](https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-ubuntu-20-04)

## Create an S3 Bucket for Workbench
Workbench pushes and pulls data from AWS, it will use this S3 Bucket for storage and processing. You should create a **NEW** S3 Bucket, we suggest a name like `<company-name>-workbench`

## Deploying the Workbench Core Stack
This stack has the `Workbench-Execution-Role` and an associated role for AWS Glue Jobs.

You'll need to set some environmental vars before deploying the stack.

```
export WORKBENCH_BUCKET=name-of-workbench-bucket
export WORKBENCH_SSO_GROUP=DataScientist (or whatever)
```

**Optional ENV Vars**

```
export WORKBENCH_ADDITIONAL_BUCKETS=<comma separated list of buckets>
```


!!! note "AWS Stuff"
    Activate your AWS Account that's used for Workbench deployment. For this one time install you should use an Admin Account (or an account that had permissions to create/update AWS Stacks)

  ```bash
  cd workbench/aws_setup/workbench_core
  pip install -r requirements.txt
  cdk bootstrap
  cdk deploy
  ```
  
### Enable Users to Assume Workbench-ExecutionRole
Now that the `Workbench-ExecutionRole` has been deployed via AWS Stack. These guides walk you through setting up access for both SSO users and IAM users to assume the Workbench-ExecutionRole in your AWS account.

- [Set up SSO Users](sso_assume_role.md)
- [Set up IAM Users](iam_assume_role.md) (not recommend, but contact us we'll help you out)


## AWS Account Setup Check
After deploying the Workbench Core Stack and setting up users to assume that Role, you can run this test/checking script. If the results ends with `INFO AWS Account Clamp: AOK!` you're in good shape. If not feel free to contact us on [Discord](https://discord.gg/WHAJuz8sw8) and we'll get it straightened out for you :)

```bash
pip install workbench (if not already installed)
cd workbench/aws_setup
python aws_account_check.py
<lot of print outs for various checks>
INFO AWS Account Clamp: AOK!
```

!!! success
    Congratulations: Workbench is now deployed to your AWS Account. Deploying the AWS Stack only needs to be done once. Now that this is complete your developers can simply `pip install workbench` and start using the API.
    
If you ran into any issues with this procedure please contact us via [Discord](https://discord.gg/WHAJuz8sw8) or email [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) and the SCP team will provide **free** setup and support for new Workbench users.
