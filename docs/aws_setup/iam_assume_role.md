# Setting Up IAM Users to Access Workbench Roles

This guide configures IAM users to assume the Workbench roles in your AWS account.

Workbench deploys three roles, and a user should be allowed to assume all three:

- **`Workbench-BuilderRole`** — the REPL's default. Full create, train, and read, but
  cannot delete or overwrite DataSources and FeatureSets.
- **`Workbench-ExecutionRole`** — full access, including delete. The REPL uses it when its
  config's `WORKBENCH_ROLE` names this role.
- **`Workbench-ReadOnlyRole`** — read-only; used by the Dashboard.

## Prerequisites

- **Administrator permissions** to update IAM users and policies.
- The Workbench roles must already be deployed via the Workbench AWS CDK stack.


## Steps to Update IAM User Permissions

### 1. Log in to the AWS Management Console

- Navigate to the [IAM Console](https://console.aws.amazon.com/iam/).

### 2. Select the IAM User

1. In the left-hand menu, select **Users**.
2. Locate and select the IAM user who needs access to the Workbench roles.

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
                "Resource": [
                    "arn:aws:iam::<account-id>:role/Workbench-BuilderRole",
                    "arn:aws:iam::<account-id>:role/Workbench-ExecutionRole",
                    "arn:aws:iam::<account-id>:role/Workbench-ReadOnlyRole"
                ]
            }
        ]
    }
    ```

    - Replace `<account-id>` with your AWS account ID.

4. Review and save the policy.


## Verifying Access for IAM Users

1. Log in to the AWS Management Console as the IAM user.
2. Use the following CLI command to test access to each role:

    ```bash
    aws sts assume-role \
        --role-arn arn:aws:iam::<account-id>:role/Workbench-BuilderRole \
        --role-session-name TestSession

    aws sts assume-role \
        --role-arn arn:aws:iam::<account-id>:role/Workbench-ExecutionRole \
        --role-session-name TestSession

    aws sts assume-role \
        --role-arn arn:aws:iam::<account-id>:role/Workbench-ReadOnlyRole \
        --role-session-name TestSession
    ```

    - Replace `<account-id>` with your AWS account ID.

3. If successful, you will receive temporary credentials for each role.


## Troubleshooting

### Common Issues

- **Permission Denied**: Ensure the correct inline policy is attached to the IAM user.
- **Role Not Found**: Verify that the Workbench roles have been deployed correctly.
- **REPL won't start / cannot assume role**: The REPL assumes `Workbench-BuilderRole` by
  default. If that role is missing from the account or the user's policy, the REPL cannot
  start — deploy the Core Stack and add the role above, or set the REPL config's
  `WORKBENCH_ROLE` to `Workbench-ExecutionRole`.

### Contact Support

If you encounter issues, please contact your AWS administrator or reach out to the Workbench support team.
