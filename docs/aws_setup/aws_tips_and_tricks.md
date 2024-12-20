# AWS Tips and Tricks
!!!tip inline end "Need AWS Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

This page tries to give helpful guidance when setting up AWS Accounts, Users, and Groups. In general AWS can be a bit tricky to set up the first time. Feel free to use any material in this guide but we're more than happy to help clients get their AWS Setup ready to go for FREE. Below are some guides for setting up a new AWS account for Workbench and also setting up SSO Users and Groups within AWS.

## New AWS Account (with AWS Organizations: easy)
- If you already have an AWS Account you can activate the AWS Identity Center/Organization functionality.
- Now go to AWS Organizations page and hit 'Add an AWS Account' button
- Add a new User with permissions that **allows AWS Stack creation**
   
!!! note inline end "Email Trick"
    AWS will often not allow the same email to be used for different accounts. If you need a 'new' email just add a plus sign '+' at the end of your existing email (e.g. bob.smith**+aws**@gmail.com). This email will 'auto forward' to bob.smith@gmail.com.


## New AWS Account (without AWS Organizations: a bit harder)
- Goto: https://aws.amazon.com/free and hit the big button 'Create a Free Account'
- Enter email and the account name you'd like (anything is fine)
- You'll get a validation email and go through the rest of the Account setup procedure
- Add a new User with permissions that **allows AWS Stack creation**


# SSO Users and Groups
AWS SSO (Single Sign-On) is a cloud-based service that allows users to manage access to multiple AWS accounts and business applications using a single set of credentials. It simplifies the authentication process for users and provides centralized management of permissions and access control across various AWS resources. With AWS SSO, users can log in once and access all the applications and accounts they need, streamlining the user experience and increasing productivity. AWS SSO also enables IT administrators to manage access more efficiently by providing a single point of control for managing user access, permissions, and policies, reducing the risk of unauthorized access or security breaches.

## Setting up SSO Users
* Log in to your AWS account and go to the AWS Identity Center console.
* Click on the "Users" tab and then click on the "Add user" button.

The 'Add User' setup is fairly straight forward but here are some screen shots:

On the first panel you can fill in the users information.

<img width="800" alt="Screenshot 2023-05-03 at 9 31 30 AM" src="https://user-images.githubusercontent.com/4806709/235965493-eaa5f879-df04-473b-b98d-03d422db7272.png">

## Groups
On the second panel we suggest that you have at LEAST two groups:

- Admin group
- DataScientists group

### Setting up Groups
This allows you to put most of the users into the DataScientists group that has AWS policies based on their job role. AWS uses 'permission sets' and you assign AWS Policies. This approach makes it easy to give a group of users a set of relevant policies for their tasks. 

Our standard setup is to have two permission sets with the following policies:

- IAM Identity Center --> Permission sets --> DataScientist 
   - Add Policy: arn:aws:iam::aws:policy/job-function/DataScientist

- IAM Identity Center --> Permission sets --> AdministratorAccess 
   - Add Policy: arn:aws:iam::aws:policy/job-function/AdministratorAccess

See: [Permission Sets](https://docs.aws.amazon.com/singlesignon/latest/userguide/permissionsetsconcept.html) for more details and instructions.

Another benefit of creating groups is that you can include that group in 'Trust Policy (assume_role)' for the Workbench-ExecutionRole (this gets deployed as part of the Workbench AWS Stack). This means that the management of what Workbench can do/see/read/write is completely done through the Workbench-ExecutionRole.

## Back to Adding User
Okay now that we have our groups set up we can go back to our original goal of adding a user. So here's the second panel with the groups and now we can hit 'Next'

<img width="800" alt="Screenshot 2023-05-03 at 9 31 49 AM" src="https://user-images.githubusercontent.com/4806709/235965818-fa44bb58-6e58-49df-93df-ba582148b3f4.png">

On the third panel just review the details and hit the 'Add User' button at the bottom. The user will get an email giving them instructions on how to log on to their AWS account.

<img width="600" alt="Screenshot 2023-05-03 at 9 32 28 AM" src="https://user-images.githubusercontent.com/4806709/235967585-d772d2f9-13ac-4795-aca3-429fbb1b7311.png">

### AWS Console
Now when the user logs onto the AWS Console they should see something like this:
<img width="800" alt="Screenshot 2023-05-03 at 9 21 27 AM" src="https://user-images.githubusercontent.com/4806709/235970829-d1fdf1a8-84a2-46ca-a20e-143664715531.png">

### AWS CLI/SSO Setup for Command Line/Python Usage
Please see our [AWS Setup](aws_setup.md)

 
## AWS Resources
- [AWS Identity Center](https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html)
- [Users and Groups](https://docs.aws.amazon.com/singlesignon/latest/userguide/users-groups-provisioning.html)
- [Permission Sets](https://docs.aws.amazon.com/singlesignon/latest/userguide/permissionsetsconcept.html)
- [SSO Command Line/Python Configure](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html)


