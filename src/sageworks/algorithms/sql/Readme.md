# Algorithms: SQL
These algorithms will be based on **[AWS Athena](https://aws.amazon.com/athena/)** or **[AWS RDS](https://aws.amazon.com/rds/)**

- **SQL:** These algorithms are composed of one or many SQL statements. 
- **Inputs:** We need to pass in a **DataSource** class handle as input.
- **Outputs** 
  - **Light:** Some algorithms will pass back light data (Python dict, or small dataframe)
  - **Heavy:** For algorithms that have add a column that will typically mean an 'in-situ' modification of the DataSource (Athena/S3 storage)
- **Heavy:** These algorithms are considered **heavy** algorithms since they are running on database that should be scalable to large datasets.