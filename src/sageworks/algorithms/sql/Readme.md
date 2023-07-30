# Algorithms: SQL
These algorithms will be based on **[AWS Athena](https://aws.amazon.com/athena/)** or **[AWS RDS](https://aws.amazon.com/rds/)**

- **SQL:** These algorithms are composed of one or many SQL statements. 
- **Inputs:** We need to pass in a **DataSource** class handle as input.
- **Outputs:** A modification of the DataSource, this modification may be of the underlying data or may be changes or updates to the meta data (details, quartiles, outliers, etc)
- **Heavy:** These algorithms are considered **heavy** algorithms since they are running on database that should be scalable to large datasets.