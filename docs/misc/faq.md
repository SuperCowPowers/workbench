# Workbench: FAQ

!!! tip question "Artifact and Column Naming?"
    You might have noticed that Workbench has some unintuitive constraints when naming Artifacts and restrictions on column names. All of these restrictions come from AWS. Workbench uses Glue, Athena, Feature Store, Models and Endpoints, each of these services have their own constraints, Workbench simply 'reflects' those contraints.
    
## Naming: Underscores, Dashes, and Lower Case

Data Sources and Feature Sets must adhere to AWS restrictions on table names and columns names (here is a snippet from the AWS documentation)

> **Database, table, and column names**
> 
> When you create schema in AWS Glue to query in Athena, consider the following:
> 
> A database name cannot be longer than 255 characters.
> A table name cannot be longer than 255 characters.
> A column name cannot be longer than 255 characters.
> 
> The only acceptable characters for database names, table names, and column names are lowercase letters, numbers, and the underscore character.

For more info see: [Glue Best Practices](https://docs.aws.amazon.com/athena/latest/ug/glue-best-practices.html#schema-names)

## DataSource/FeatureSet use '_'  and Model/Endpoint use '-'

You may notice that DataSource and FeatureSet uuid/name examples have underscores but the model and endpoints have dashes. Yes, it’s super annoying to have one convention for DataSources and FeatureSets and another for Models and Endpoints but this is an AWS restriction and not something that Workbench can control.

**DataSources and FeatureSet:** Underscores. You cannot use a dash because both classes use Athena for Storage and Athena tables names cannot have a dash.

**Models and Endpoints:** Dashes. You cannot use an underscores because AWS imposes a restriction on the naming.


## Additional information on the lower case issue
We’ve tried to create a glue table with Mixed Case column names and haven’t had any luck. We’ve bypassed wrangler and used the boto3 low level calls directly. In all cases when it shows up in the Glue Table the columns have always been converted to lower case. We've also tried uses the Athena DDL directly, that also doesn't work. Here's the relevant AWS documentation and the two scripts that reproduce the issue.

**AWS Docs**

- [Athena Naming Restrictions](https://docs.aws.amazon.com/athena/latest/ug/tables-databases-columns-names.html)
- [Glue Best Practices](https://docs.aws.amazon.com/athena/latest/ug/glue-best-practices.html#schema-names)

**Scripts to Reproduce**

- [scripts/athena\_ddl\_mixed_case.py](https://github.com/SuperCowPowers/workbench/blob/main/scripts/athena_ddl_mixed_case.py)
- [scripts/glue\_mixed_case.py](https://github.com/SuperCowPowers/workbench/blob/main/scripts/glue_mixed_case.py)

