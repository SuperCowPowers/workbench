# Workbench Windows Box

A maintainer test box: a Windows Server 2025 instance you start for a session of
hands-on Windows testing and stop afterwards. Setup done on the box (AWS profile,
Workbench config, venvs) persists across stops.

- Default VPC, no inbound rules; RDP runs through an SSM port forward.
- First boot installs python.org Python 3.13 (on PATH) and the AWS CLI. Long paths
  stay off, the Windows default.
- A CloudWatch alarm stops the instance after 2 hours below 5% CPU.
- Cost: ~$0.11/hour running (t3.large, Windows license included), ~$5/month stopped
  (60 GB gp3).

## Deploy

```bash
cd workbench/aws_setup/workbench_windows_box
pip install -r requirements.txt
cdk deploy
```

The outputs give the `InstanceId` and the `KeyParameter` (SSM parameter holding the
private key). The AMI is cached in `cdk.context.json`, so a redeploy keeps the same
instance; clearing that context picks up a newer image and replaces the instance.

## Administrator password (once)

The password is ready a few minutes after the first boot.

```bash
aws ssm get-parameter --name <KeyParameter> --with-decryption --query Parameter.Value --output text > windows-box.pem
aws ec2 get-password-data --instance-id <InstanceId> --priv-launch-key windows-box.pem --query PasswordData --output text
```

## Each session

```bash
aws ec2 start-instances --instance-ids <InstanceId>
aws ssm start-session --target <InstanceId> --document-name AWS-StartPortForwardingSession --parameters portNumber=3389,localPortNumber=13389
```

Connect with Microsoft's Windows App to `localhost:13389` as `Administrator`. The
port forward needs the Session Manager plugin (`brew install --cask session-manager-plugin`).
When done (or let the idle alarm do it):

```bash
aws ec2 stop-instances --instance-ids <InstanceId>
```

## On the box (first session)

In PowerShell:

```powershell
aws configure sso                      # creates the AWS profile
mkdir $HOME\.workbench                 # copy your Workbench config JSON here over RDP
setx AWS_PROFILE <profile>
setx WORKBENCH_CONFIG "$HOME\.workbench\<config>.json"
```

Open a new terminal so the variables apply, then install and test:

```powershell
pip install workbench
pip install https://github.com/SuperCowPowers/workbench/archive/refs/heads/main.zip   # unreleased main
workbench
```

## Teardown

```bash
cdk destroy
```
