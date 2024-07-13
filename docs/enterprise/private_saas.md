# Benefits of a Private SaaS Architecture


### Self Hosted vs Private SaaS vs Public SaaS?
At the top level your team/project is making a decision about how they are going to build, expand, support, and maintain a machine learning pipeline.

**Conceptual ML Pipeline**

```
Data ⮕ Features ⮕ Models ⮕ Deployment (end-user application)
```


**Concrete/Real World Example**

```
S3 ⮕ Glue Job ⮕ Data Catalog ⮕ FeatureGroups ⮕ Models ⮕ Endpoints ⮕ App
```

When building out a framework to support ML Pipelines there are three main options:

- **Self Hosted**
- **Private SaaS**
- **Public SaaS**

The other choice, that we're not going to cover here, is whether you use AWS, Azure, GCP, or something else. SageWorks is architected and powered by a broad and rich set of AWS ML Pipeline services. We believe that AWS provides the best set of functionality and APIs for flexible, real world ML architectures.

<img alt="private_saas_compare" src="https://github.com/user-attachments/assets/2f6d3724-e340-4a70-bb97-d05383917cfe">


### Resources
See our full presentation on the SageWorks [Private SaaS Architecture](https://docs.google.com/presentation/d/1f_1gmE4-UAeUDDsoNdzK_d_MxALFXIkxORZwbJBjPq4/edit?usp=sharing``)