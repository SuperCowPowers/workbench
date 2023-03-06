# SageWorks Artifacts

All stored entities* in SageWorks are called Artifacts. There is an `artifact` superclass that specifies a small API used by all subclasses. Note: We might want to make this a Python dataclass.

**Artifact API**

- `name() -> str:` Name of the artifact.
- `meta() -> dict:` Meta data associated with this artifact.
- `size() -> int:` Size of data in MegaBytes
- `tags() -> list:` Tags associated with the Artifact
- `date_created() -> datetime:` Creation date
- `date_modified() -> datetime:` Last modification date


\* Stored Entities = Stored in AWS Services like Data Catalog, Feature Store, Model Registry, etc.