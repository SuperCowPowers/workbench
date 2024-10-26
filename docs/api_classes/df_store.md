# SageWorks DataFrame Storage

!!! tip inline end "Examples"
    Examples of using the Parameter Storage class are listed at the bottom of this page [Examples](#examples).
    
## Why DataFrame Storage?
Great question, there's a couple of reasons. The first is that the Parameter Store in AWS has a 4KB limit, so that won't support any kind of 'real data'. The second reason is that DataFrames are commonly used as part of the data engineering, science, and ML pipeline construction process. Providing storage of **named** DataFrames in an accessible location that can be inspected and used by your ML Team comes in super handy.

## Efficient Storage
All DataFrames are stored in the Parquet format using 'snappy' storage. Parquet is a columnar storage format that efficiently handles large datasets, and using Snappy compression reduces file size while maintaining fast read/write speeds.
    
::: sageworks.api.df_store


## Examples
These example show how to use the `DFStore()` class to list, add, and get dataframes from AWS Storage.

!!!tip "SageWorks REPL"
    If you'd like to experiment with listing, adding, and getting dataframe with the `DFStore()` class, you can spin up the SageWorks REPL, use the class and test out all the methods. Try it out! [SageWorks REPL](../repl/index.md)

```py title="Using DataFrame Store"
from sageworks.api.df_store import DFStore
df_store = DFStore()

# List DataFrames
df_store().list()

Out[1]:
ml/confustion_matrix  (0.002MB/2024-09-23 16:44:48)
ml/hold_out_ids  (0.094MB/2024-09-23 16:57:01)
ml/my_awesome_df  (0.002MB/2024-09-23 16:43:30)
ml/shap_values  (0.019MB/2024-09-23 16:57:21)
 
# Add a DataFrame
df = pd.DataFrame({"A": [1]*1000, "B": [3]*1000})
df_store.upsert("test/test_df", df)

# List DataFrames (we can just use the REPR)
df_store

Out[2]:
ml/confustion_matrix  (0.002MB/2024-09-23 16:44:48)
ml/hold_out_ids  (0.094MB/2024-09-23 16:57:01)
ml/my_awesome_df  (0.002MB/2024-09-23 16:43:30)
ml/shap_values  (0.019MB/2024-09-23 16:57:21)
test/test_df  (0.002MB/2024-09-23 16:59:27)

# Retrieve dataframes
return_df = df_store.get("test/test_df")
return_df.head()

Out[3]:
   A  B
0  1  3
1  1  3
2  1  3
3  1  3
4  1  3

# Delete dataframes
df_store.delete("test/test_df")
```

!!! note "Compressed Storage is Automatic"
    All DataFrames are stored in the Parquet format using 'snappy' storage. Parquet is a columnar storage format that efficiently handles large datasets, and using Snappy compression reduces file size while maintaining fast read/write speeds.