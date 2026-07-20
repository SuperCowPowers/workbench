# Setting Up SSO Users to Access Workbench Roles

This guide configures AWS SSO users to assume the Workbench roles in your AWS account.

Workbench deploys three roles, and a user's permission set should allow all three:

- **`Workbench-BuilderRole`** — the REPL's default. Full create, train, and read, but
  cannot delete or overwrite DataSources and FeatureSets.
- **`Workbench-ExecutionRole`** — full access, including delete. The REPL uses it when its
  config's `WORKBENCH_ROLE` names this role.
- **`Workbench-ReadOnlyRole`** — read-only; used by the Dashboard.

## Prerequisites

- Access to the **management account** in your AWS Organization.
- **Administrative permissions** to modify AWS SSO permission sets.
- The Workbench roles must already be deployed via the Workbench AWS CDK stack.


## Steps to Update SSO Permissions

### 1. Log in to the AWS SSO Console

- Login to your 'main' organization AWS Account.
- Go to the IAM Identity Center

### 2. Find the Relevant Permission Set

- In the left menu, select **Permission Sets**.
- Locate the permission set used by the group needing access (e.g., `DataScientist` or another relevant group).

### 3. Edit the Permission Set

1. Select the permission set
2. Scroll down to **Inline policy** and click the Edit button.
3. Add an **inline policy** with the following content:

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

Replace `<account-id>` with your AWS account ID. For SSO Groups that span multiple
accounts, list all three roles for each account:

```json
"Resource": [
    "arn:aws:iam::<account-1>:role/Workbench-BuilderRole",
    "arn:aws:iam::<account-1>:role/Workbench-ExecutionRole",
    "arn:aws:iam::<account-1>:role/Workbench-ReadOnlyRole",

    "arn:aws:iam::<account-2>:role/Workbench-BuilderRole",
    "arn:aws:iam::<account-2>:role/Workbench-ExecutionRole",
    "arn:aws:iam::<account-2>:role/Workbench-ReadOnlyRole"
]
```

**Please consult with your AWS SSO Administrator for guidance on this process.**

### 4. Save Changes

- Save the updated permission set.
- AWS SSO will automatically propagate the changes to all users in the associated group.


## Verifying Access for SSO Users

1. Activate an AWS Profile for the configured SSO group.
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

    Replace `<account-id>` with your AWS account ID.

3. If successful, you will receive temporary credentials.


## Troubleshooting

### Common Issues

- **Permission Denied**: Ensure the correct permission set is updated.
- **Role Not Found**: Verify that the Workbench roles have been deployed correctly.
- **REPL won't start / cannot assume role**: The REPL assumes `Workbench-BuilderRole` by
  default. If that role is missing from the account or the permission set, the REPL cannot
  start — deploy the Core Stack and add the role above, or set the REPL config's
  `WORKBENCH_ROLE` to `Workbench-ExecutionRole`.

### Contact Support

If you encounter issues, please contact your AWS administrator or reach out to the Workbench support team.
