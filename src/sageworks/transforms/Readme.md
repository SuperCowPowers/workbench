# SageWorks Transforms

All clases that tranform an Artifact (stored entity*) to another Artifact is called a **Transform**. There is an `trasform` superclass that specifies a small API used by all subclasses. 

**Transform API**

- `input_type() -> enum:` Get the input artifact type.
- `output_type() -> enum:` Get the output artifact type.
- `validate_input() -> enum:`Valdate the given input artifact. Correct type? Can we reach it? Is it too big?
- `validate_output() -> enum:`Valdate the given output artifact AFTER we create it. Can we query it?

\* Stored Entity = Stored in AWS Services like Data Catalog, Feature Store, Model Registry, etc.