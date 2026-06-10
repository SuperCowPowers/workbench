# GitHub Actions → AWS (OIDC Setup)

A start-to-finish guide for wiring GitHub Actions to an AWS account so the
[AWS Tests workflow](index.md) can run `tox -m quick` against real
infrastructure — without storing AWS keys in GitHub. Written so it can be
repeated for a new account/customer from scratch.

**Auth chain:** GitHub OIDC token → `github-actions-workbench` role (federated,
short-lived credentials) → `sts:AssumeRole` → `Workbench-ExecutionRole`. The
CI ends up using the exact same role workbench uses everywhere else, so no new
permission set is defined or maintained.

Replace `<AWS_ACCOUNT_ID>` with the target account ID and `<org>/<repo>` with
the GitHub repository that runs the workflow (for workbench itself:
`SuperCowPowers/workbench`).

## Prerequisites

- Admin credentials for the target AWS account (an `<account>_admin` AWS
  profile with rights to deploy the core stack and create IAM roles)
- The account's Workbench admin config (e.g. `~/.workbench/<account>_admin.json`)
- Admin access to the GitHub repository (to create secrets)
- `aws` CLI and `cdk` installed; `gh` CLI optional but handy for secrets

## 1. Trust the CI role in the core stack

The Workbench core stack supports an optional `WORKBENCH_TRUSTED_ARNS` config —
comma-separated role ARNs appended to the Workbench roles' trust policies.
When unset, the stack is unchanged, so this is safe to leave out of every
other account's config.

Add to the account's admin config file:

```json
"WORKBENCH_TRUSTED_ARNS": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/github-actions-workbench"
```

Then diff and deploy:

```bash
cd aws_setup/workbench_core
WORKBENCH_CONFIG=~/.workbench/<account>_admin.json cdk diff
WORKBENCH_CONFIG=~/.workbench/<account>_admin.json cdk deploy
```

The diff should show exactly one kind of change: the new ARN appearing in the
`AssumeRolePolicyDocument` condition list on `Workbench-ExecutionRole` and
`Workbench-ReadOnlyRole`. If anything else shows up, stop and investigate.

> Note: the role ARN can be referenced in the trust policy before the role
> exists (step 3) — IAM trust conditions on `aws:PrincipalArn` are just string
> patterns. Order between steps 1 and 3 doesn't matter.

## 2. Create the GitHub OIDC provider

One per account (shared by all repos/workflows). Check first:

```bash
aws iam list-open-id-connect-providers
```

If `token.actions.githubusercontent.com` isn't listed:

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com
```

## 3. Create the CI role

The trust policy is the security boundary: the `sub` condition means only
workflows from the named repo's `main` branch can assume this role. Fork PRs,
other branches, and other repos are rejected by AWS before any permissions
are evaluated.

```bash
aws iam create-role --role-name github-actions-workbench \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Federated": "arn:aws:iam::<AWS_ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"},
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
        "StringLike": {"token.actions.githubusercontent.com:sub": "repo:<org>/<repo>:ref:refs/heads/main"}
      }
    }]
  }'
```

Useful `sub` variations:

| Pattern | Allows |
| --- | --- |
| `repo:<org>/<repo>:ref:refs/heads/main` | `main` branch only (recommended) |
| `repo:<org>/<repo>:ref:refs/tags/v*` | version tags |
| `repo:<org>/<repo>:*` | any branch/PR in the repo (avoid — includes fork PR merge refs on some triggers) |

## 4. Grant its only permission

The CI role carries no workbench permissions itself — just the right to become
the execution role:

```bash
aws iam put-role-policy --role-name github-actions-workbench \
  --policy-name assume-workbench-execution-role \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Action": "sts:AssumeRole",
      "Resource": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/Workbench-ExecutionRole"}]
  }'
```

## 5. GitHub repository secrets

The workflow ([`aws-tests.yml`](https://github.com/SuperCowPowers/workbench/blob/main/.github/workflows/aws-tests.yml))
reads three secrets:

| Secret | Value | Where it comes from |
| --- | --- | --- |
| `AWS_ACCOUNT_SANDBOX` | 12-digit AWS account ID | `aws sts get-caller-identity` |
| `WORKBENCH_BUCKET_SANDBOX` | Workbench artifact bucket name | `WORKBENCH_BUCKET` in the account's workbench config |
| `WORKBENCH_API_KEY_SANDBOX` | Workbench API key | `WORKBENCH_API_KEY` in the account's workbench config |

**Via the GitHub UI:** repo → *Settings* → *Secrets and variables* →
*Actions* → *New repository secret*.

**Via the `gh` CLI** (prompts for each value; nothing lands in shell history):

```bash
gh secret set AWS_ACCOUNT_SANDBOX --repo <org>/<repo>
gh secret set WORKBENCH_BUCKET_SANDBOX --repo <org>/<repo>
gh secret set WORKBENCH_API_KEY_SANDBOX --repo <org>/<repo>
```

At runtime, `WORKBENCH_BUCKET` / `WORKBENCH_API_KEY` are injected as env vars,
which override the placeholder values in the checked-in
[`ci/workbench_ci_config.json`](https://github.com/SuperCowPowers/workbench/blob/main/ci/workbench_ci_config.json) —
so no account-specific values are committed to the repo.

## 6. Verify end-to-end

Trigger the workflow manually: repo → *Actions* → *AWS Tests* →
*Run workflow* (or `gh workflow run aws-tests.yml`).

Watch the **Configure AWS credentials** step — success there proves the OIDC
provider, trust policy, and secrets are all correct. The test run itself then
proves the `Workbench-ExecutionRole` assumption works (workbench logs
`AWS Credentials Refreshed` when it assumes the role).

## Troubleshooting

| Error | Likely cause |
| --- | --- |
| `Not authorized to perform sts:AssumeRoleWithWebIdentity` | OIDC provider missing (step 2), or the trust policy's `sub` pattern doesn't match the repo/branch running the workflow (step 3) |
| `Credentials could not be loaded` at the configure step | `id-token: write` permission missing in the workflow, or `AWS_ACCOUNT_SANDBOX` secret unset (the role ARN comes out malformed) |
| `AccessDenied` on `sts:AssumeRole` for `Workbench-ExecutionRole` | Core stack trust change not deployed (step 1), or the CI role's inline policy missing (step 4) |
| Workbench exits with `AWS SSO Token Failure` | `WORKBENCH_CONFIG` not pointing at `ci/workbench_ci_config.json`, so workbench picked up a profile-based config |
