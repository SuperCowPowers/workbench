# Run the SageWorks Dashboard

You can run the Dashboard on a local environment:

```bash 
$ cd applications/aws_dashboard
$ pip install -r requirements.txt
$ python app.py  (or ./dashboard)
``` 

## Docker and Getting SSO Token
In general you should talk to one of the SageWorks developers if you want to do local testing with docker. This is a development corner case, and 99.9% of folks won't need this.

But just for reference:

```
aws sts assume-role --role-arn  
arn:aws:iam::<account_nam>:role/SageWorks-ExecutionRole 
--role-session-name "DockerSession"
```
Take the output from this and fill in the command below

```
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=<AccessKeyId> \
  -e AWS_SECRET_ACCESS_KEY=<SecretAccessKey> \
  -e AWS_SESSION_TOKEN=<SessionToken> \
  -e AWS_DEFAULT_REGION=us-west-2 \
  sageworks_dashboard
```