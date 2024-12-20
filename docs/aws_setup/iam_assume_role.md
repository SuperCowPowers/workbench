# Setting Up IAM Users to Access Workbench-ExecutionRole

This guide provides step-by-step instructions to configure IAM users to assume the `Workbench-ExecutionRole` in your AWS account.

## Prerequisites

- **Administrator permissions** to update IAM users and policies.
- The `Workbench-ExecutionRole` must already be deployed via the Workbench AWS CDK stack.


## Steps to Update IAM User Permissions

### 1. Log in to the AWS Management Console

- Navigate to the [IAM Console](https://console.aws.amazon.com/iam/).

### 2. Select the IAM User

1. In the left-hand menu, select **Users**.
2. Locate and select the IAM user who needs access to the `Workbench-ExecutionRole`.

### 3. Attach an Inline Policy

1. Navigate to the **Permissions** tab for the IAM user.
2. Click **Add inline policy**.
3. Select the **JSON** editor and paste the following policy:

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

4. Review and save the policy.


## Verifying Access for IAM Users

1. Log in to the AWS Management Console as the IAM user.
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

- **Permission Denied**: Ensure the correct inline policy is attached to the IAM user.
- **Role Not Found**: Verify that the `Workbench-ExecutionRole` has been deployed correctly.

### Contact Support

If you encounter issues, please contact your AWS administrator or reach out to the Workbench support team.
