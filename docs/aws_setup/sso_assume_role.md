# Setting Up SSO Users to Access Workbench-ExecutionRole

This guide provides step-by-step instructions to configure AWS SSO users to assume the `Workbench-ExecutionRole` in your AWS account.

## Prerequisites

- Access to the **management account** in your AWS Organization.
- **Administrative permissions** to modify AWS SSO permission sets.
- The `Workbench-ExecutionRole` must already be deployed via the Workbench AWS CDK stack.


## Steps to Update SSO Permissions

### 1. Log in to the AWS SSO Console

- Login to your 'main' organization AWS Account.
- Go to the IAM Identity Center

### 2. Find the Relevant Permission Set

- In the left menu, select **Permission Sets**.
- Locate the permission set used by the group needing access (e.g., `DataScientist` or another relevant group).

### 3. Edit the Permission Set

1. Select the permission set
2. Scroll down to **Inline policy** and click the the Edit button.
2. Add an **inline policy** with the following content:

    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sts:AssumeRole",
                "Resource": "arn:aws:iam::<account-id>:role/Workbench-ExecutionRole"
            }
        ]
    }
    ```

    - Replace `<account-id>` with your AWS account ID.

### 4. Save Changes

- Save the updated permission set.
- AWS SSO will automatically propagate the changes to all users in the associated group.


## Verifying Access for SSO Users

1. Log in to the AWS Management Console as a user in the configured group.
2. Use the following CLI command to test access:

    ```bash
    aws sts assume-role \
        --role-arn arn:aws:iam::<account-id>:role/Workbench-ExecutionRole \
        --role-session-name TestSession
    ```

    - Replace `<account-id>` with your AWS account ID.

3. If successful, you will receive temporary credentials for the `Workbench-ExecutionRole`.


## Troubleshooting

### Common Issues

- **Permission Denied**: Ensure the correct permission set is updated.
- **Role Not Found**: Verify that the `Workbench-ExecutionRole` has been deployed correctly.

### Contact Support

If you encounter issues, please contact your AWS administrator or reach out to the Workbench support team.