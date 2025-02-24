# Chemical Utilities
!!! tip inline end "Endpoint Examples"
    Examples of using the Endpoint class are listed at the bottom of this page [Examples](#examples).

The majority of the chemical utilities in Workbench use either RDKIT or Mordred (Community). The inclusion of these utilities allows the use and deployment of this functionality into AWS (FeatureSets, Models, Endpoints). 
    
::: workbench.utils.chem_utils


## Examples

### Canonical Smiles

```py title="examples/chem_utils/canonicalize_smiles.py"
"""Example for computing Canonicalize SMILES strings"""
import pandas as pd
from workbench.utils.chem_utils import canonicalize

test_data = [
    {"id": "Acetylacetone", "smiles": "CC(=O)CC(=O)C", "expected": "CC(=O)CC(C)=O"},
    {"id": "Imidazole", "smiles": "c1cnc[nH]1", "expected": "c1c[nH]cn1"},
    {"id": "Pyridone", "smiles": "C1=CC=NC(=O)C=C1", "expected": "O=c1cccccn1"},
    {"id": "Guanidine", "smiles": "C(=N)N=C(N)N", "expected": "N=CN=C(N)N"},
    {"id": "Catechol", "smiles": "c1cc(c(cc1)O)O", "expected": "Oc1ccccc1O"},
    {"id": "Formamide", "smiles": "C(=O)N", "expected": "NC=O"},
    {"id": "Urea", "smiles": "C(=O)(N)N", "expected": "NC(N)=O"},
    {"id": "Phenol", "smiles": "c1ccc(cc1)O", "expected": "Oc1ccccc1"},
]

# Convert test data to a DataFrame
df = pd.DataFrame(test_data)

# Perform canonicalization
result_df = canonicalize(df)
print(result_df)
```

**Output**

```py
              id            smiles       expected smiles_canonical
0  Acetylacetone     CC(=O)CC(=O)C  CC(=O)CC(C)=O    CC(=O)CC(C)=O
1      Imidazole        c1cnc[nH]1     c1c[nH]cn1       c1c[nH]cn1
2       Pyridone  C1=CC=NC(=O)C=C1    O=c1cccccn1      O=c1cccccn1
3      Guanidine      C(=N)N=C(N)N     N=CN=C(N)N       N=CN=C(N)N
4       Catechol    c1cc(c(cc1)O)O     Oc1ccccc1O       Oc1ccccc1O
5      Formamide            C(=O)N           NC=O             NC=O
6           Urea         C(=O)(N)N        NC(N)=O          NC(N)=O
7         Phenol       c1ccc(cc1)O      Oc1ccccc1        Oc1ccccc1
```

### Tautomerize Smiles
```py title="examples/chem_utils/tautomerize_smiles.py"
"""Example for Tautomerizing SMILES strings"""
import pandas as pd
from workbench.utils.chem_utils import tautomerize_smiles

test_data = [
    # Salicylaldehyde undergoes keto-enol tautomerization.
    {"id": "Salicylaldehyde (Keto)", "smiles": "O=Cc1cccc(O)c1", "expected": "O=Cc1cccc(O)c1"},
    {"id": "2-Hydroxybenzaldehyde (Enol)", "smiles": "Oc1ccc(C=O)cc1", "expected": "O=Cc1ccc(O)cc1"},
    # Acetylacetone undergoes keto-enol tautomerization to favor the enol form.
    {"id": "Acetylacetone", "smiles": "CC(=O)CC(=O)C", "expected": "CC(=O)CC(C)=O"},
    # Imidazole undergoes a proton shift in the aromatic ring.
    {"id": "Imidazole", "smiles": "c1cnc[nH]1", "expected": "c1c[nH]cn1"},
    # Pyridone prefers the lactam form in RDKit's tautomer enumeration.
    {"id": "Pyridone", "smiles": "C1=CC=NC(=O)C=C1", "expected": "O=c1cccccn1"},
    # Guanidine undergoes amine-imine tautomerization.
    {"id": "Guanidine", "smiles": "C(=N)N=C(N)N", "expected": "N=C(N)N=CN"},
    # Catechol standardizes hydroxyl group placement in the aromatic system.
    {"id": "Catechol", "smiles": "c1cc(c(cc1)O)O", "expected": "Oc1ccccc1O"},
    # Formamide canonicalizes to NC=O, reflecting its stable form.
    {"id": "Formamide", "smiles": "C(=O)N", "expected": "NC=O"},
    # Urea undergoes a proton shift between nitrogen atoms.
    {"id": "Urea", "smiles": "C(=O)(N)N", "expected": "NC(N)=O"},
    # Phenol standardizes hydroxyl group placement in the aromatic system.
    {"id": "Phenol", "smiles": "c1ccc(cc1)O", "expected": "Oc1ccccc1"}
]

# Convert test data to a DataFrame
df = pd.DataFrame(test_data)

# Perform tautomerization
result_df = tautomerize_smiles(df)
print(result_df)
```

**Output**

```
                             id       smiles_orig        expected smiles_canonical          smiles
0        Salicylaldehyde (Keto)    O=Cc1cccc(O)c1  O=Cc1cccc(O)c1   O=Cc1cccc(O)c1  O=Cc1cccc(O)c1
1  2-Hydroxybenzaldehyde (Enol)    Oc1ccc(C=O)cc1  O=Cc1ccc(O)cc1   O=Cc1ccc(O)cc1  O=Cc1ccc(O)cc1
2                 Acetylacetone     CC(=O)CC(=O)C   CC(=O)CC(C)=O    CC(=O)CC(C)=O   CC(=O)CC(C)=O
3                     Imidazole        c1cnc[nH]1      c1c[nH]cn1       c1c[nH]cn1      c1c[nH]cn1
4                      Pyridone  C1=CC=NC(=O)C=C1     O=c1cccccn1      O=c1cccccn1     O=c1cccccn1
5                     Guanidine      C(=N)N=C(N)N      N=C(N)N=CN       N=CN=C(N)N      N=C(N)N=CN
6                      Catechol    c1cc(c(cc1)O)O      Oc1ccccc1O       Oc1ccccc1O      Oc1ccccc1O
7                     Formamide            C(=O)N            NC=O             NC=O            NC=O
8                          Urea         C(=O)(N)N         NC(N)=O          NC(N)=O         NC(N)=O
9                        Phenol       c1ccc(cc1)O       Oc1ccccc1        Oc1ccccc1       Oc1ccccc1
```



## Additional Resources

<img align="right" src="../images/scp.png" width="180">

- Workbench API Classes: [API Classes](../api_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
