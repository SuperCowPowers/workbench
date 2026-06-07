# AWS Setup Overview

Workbench runs on your own AWS account. Setting it up is mostly a **one-time
administrator task**; individual users do almost nothing.

!!! tip "Need AWS help?"
    The SuperCowPowers team is happy to help set up AWS and Workbench for
    **free** — [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
    or on [Discord](https://discord.gg/WHAJuz8sw8).

## Who does what

**Administrators** — done **once per AWS account**: create the account/users,
deploy the [Workbench stacks](stacks.md), and grant role access. Follow the
*Administrator Setup* pages in order.

**Users** — done by **each person**: configure an AWS CLI/SSO profile
([AWS CLI/SSO Setup](aws_setup.md)), then `pip install 'workbench[all]'` and use
the API. That's it — users never deploy stacks.

## Administrator setup flow

Do these in order (the optional stacks can be skipped or added later):

1. [AWS Account & Users](aws_tips_and_tricks.md) — create the account, SSO users, and groups
2. [Core Stack](core_stack.md) — deploy the foundational `Workbench-ExecutionRole` (**required**)
3. Grant role access — [SSO Users](sso_assume_role.md) or [IAM Users](iam_assume_role.md)
4. *(optional)* [Compute Stack](compute_stack.md) — AWS Batch for pipeline fan-out
5. *(optional)* [Dashboard Stack](dashboard_stack.md) — the team web UI (+ optional [domain/SSL](domain_cert_setup.md))
6. [Verify](full_pipeline.md) — build a full ML pipeline end-to-end as a smoke test

See [The Workbench Stacks](stacks.md) for what each stack provisions and when you need it.

## Useful AWS references

- [AWS Identity Center](https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html)
- [Users and Groups](https://docs.aws.amazon.com/singlesignon/latest/userguide/users-groups-provisioning.html)
- [Permission Sets](https://docs.aws.amazon.com/singlesignon/latest/userguide/permissionsetsconcept.html)
- [SSO Command Line/Python Configure](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html)
