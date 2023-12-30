# Initial AWS Setup
Welcome to the SageWorks AWS Setup Guide. SageWorks is deployed as an AWS **Stack** following the well architected system practices of AWS. 

!!! warning "AWS Setup can be a bit complex"
    Setting up SageWorks with AWS can be a bit complex, but you only have to do it ONCE and SageWorks tries to make it straight forward. If you have any troubles at all feel free to contact us a [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on [Discord](https://discord.gg/WHAJuz8sw8) and we're happy to help you with AWS for FREE.
    
## Two main options when using SageWorks
1. Spin up a new AWS Account for the SageWorks Stacks ([Make a New Account](aws_tips_and_tricks.md))
2. Deploy SageWorks Stacks into your existing AWS Account

Either of these options are fully supported, but we highly suggest a NEW account as it gives the following benefits:

- **AWS Data Isolation:** Data Scientists will feel empowered to play in the sandbox without impacting production services.
- **AWS Cost Accounting:** Monitor and Track all those new ML Pipelines that your team creates with SageWorks :)

## Setting up Users and Groups
If your AWS Account already has users and groups set up you can skip this but here's our recommendations on setting up [SSO Users and Groups](aws_tips_and_tricks.md)

## Onboarding SageWorks to your AWS Account

Pulling down the SageWorks Repo
  ```
  git clone https://github.com/SuperCowPowers/sageworks.git
  ```

## SageWorks uses AWS Python CDK for Deployments
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

## Create an S3 Bucket for SageWorks
SageWorks pushes and pulls data from AWS, it will use this S3 Bucket for storage and processing. You should create a **NEW** S3 Bucket, we suggest a name like `<company_name/url>-sageworks`

## Deploying the SageWorks Core Stack

!!! note "AWS Stuff"
    Activate your AWS Account that's used for SageWorks deployment. For this one time install you should use an Admin Account (or an account that had permissions to create/update AWS Stacks)

  ```bash
  cd sageworks/aws_setup/sageworks_core
  export AWS_PROFLE=<aws_admin_account>
  export SAGEWORKS_BUCKET=<name of your S3 bucket>
  (optional) export SAGEWORKS_SSO_GROUP=<your SSO group>
  pip install -r requirements.txt
  cdk bootstrap
  cdk deploy
  ```

## AWS Account Setup Check
After setting up SageWorks config/AWS Account you can run this test/checking script. If the results ends with `INFO AWS Account Clamp: AOK!` you're in good shape. If not feel free to contact us on [Discord](https://discord.gg/WHAJuz8sw8) and we'll get it straightened out for you :)

```bash
pip install sageworks
cd sageworks/aws_setup
python aws_account_check.py
<lot of print outs for various checks>
2023-04-12 11:17:09 (aws_account_check.py:48) INFO AWS Account Clamp: AOK!
```

## Building our first ML Pipeline
Okay, now the more significant testing. We're literally going to build an entire AWS ML Pipeline. The script `build_ml_pipeline.py` uses the SageWorks API to quickly and easily build an AWS Modeling Pipeline.
- DataLoader(abalone.csv) --> DataSource
- DataToFeatureSet Transform --> FeatureSet
- FeatureSetToModel Transform --> Model
- ModelToEndpoint Transform --> Endpoint

This script will take a LONG TiME to run, most of the time is waiting on AWS to finalize FeatureGroup population.

```
❯ python build_ml_pipeline.py
<lot of building ML pipeline outputs>
```
After the script completes you will see that it's built out an AWS ML Pipeline and testing artifacts.

## Run the SageWorks Dashboard (Local)
!!! tip inline end "Dashboard AWS Stack"
    Deploying the Dashboard Stack is straight-forward and provides a robust AWS Web Server with Load Balancer, Elastic Container Service, VPC Networks, etc. (see [AWS Dashboard Stack](dashboard_stack.md))

For testing it's nice to run the Dashboard locally, but for longterm use the SageWorks Dashboard should be deployed as an AWS Stack. The deployed Stack allows everyone in the company to use, view, and interact with the AWS Machine Learning Artifacts created with SageWorks.

```
cd sageworks/application/aws_dashboard
./dashboard
```
**This will open a browser to http://localhost:8000**

<figure">
<img alt="sageworks_new_light" src="https://github.com/SuperCowPowers/sageworks/assets/4806709/5f8b32a2-ed72-45f2-bd96-91b7bbbccff4">
<figcaption>SageWorks Dashboard: AWS Pipelines in a Whole New Light!</figcaption>
</figure>


!!! success
    Congratulations: SageWorks is now deployed to your AWS Account. Deploying the AWS Stack only needs to be done once. Now that this is complete your developers can simply `pip install sageworks` and start using the API.
    
If you ran into any issues with this procedure please contact us via [Discord](https://discord.gg/WHAJuz8sw8) or email [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) and the SCP team will provide **free** setup and support for new SageWorks users.
