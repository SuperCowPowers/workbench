# SageWorks Parameter Storage

!!! tip inline end "Examples"
    Examples of using the Parameter Storage class are listed at the bottom of this page [Examples](#examples).
    
::: sageworks.api.parameter_store

## Examples
These example show how to use the `ParameterStore()` class to list, add, and get parameters from the AWS Parameter Store Service.

!!!tip "SageWorks REPL"
    If you'd like to experiment with listing, adding, and getting data with the `ParameterStore()` class, you can spin up the SageWorks REPL, use the class and test out all the methods. Try it out! [SageWorks REPL](../repl/index.md)

```py title="Using SageWorks REPL"
params = ParameterStore()
params.list()
params.add("key", "value")
value = params.get("key")

# Can also store lists and dictionaries
params.add("my_data", {"key": "value", "number": 4.2, "list": [1,2,3]})
return_value = params.get("my_data")
pprint(return_value)

{'key': 'value', 'list': [1, 2, 3], 'number': 4.2}

# Delete parameters
param_store.delete("my_data")
```


!!! note "List() not showing ALL parameters?"
    If you want access to ALL the parameters in the parameter store set `prefix=None` and everything will show up.

    ```
    params = ParameterStore(prefix=None)
    params.list()
    <all the keys>
    ```
