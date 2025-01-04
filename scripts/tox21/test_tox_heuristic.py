from rdkit import Chem
from workbench.api.df_store import DFStore
from workbench.utils.chem_utils import toxic_elements, toxic_groups

# Grab the tox21 data
tox21 = DFStore().get("/datasets/chem_info/tox21")

# Add molecules to the dataframe
tox21["molecule"] = tox21["smiles"].apply(lambda smiles: Chem.MolFromSmiles(smiles))

# See which molecules are tagged as toxic but not toxic (based on the tox21 dataset)
too_broad = tox21[(tox21["toxic_tag"] == 1) & (tox21["toxic_any"] == 0)].copy()

# Let's check to see if the too_broad molecules still get flagged as toxic
too_broad["toxic_elements"] = too_broad["molecule"].apply(toxic_elements)
too_broad["toxic_groups"] = too_broad["molecule"].apply(toxic_groups)

# Filter out ones with known toxic groups
toxic_patterns = ["[N+](=O)[O-]", "C#N", "[C](Cl)(Cl)Cl"]  # Known toxic SMARTS patterns
too_broad = too_broad[
    ~too_broad["smiles"].apply(
        lambda s: any(Chem.MolFromSmiles(s).HasSubstructMatch(Chem.MolFromSmarts(p)) for p in toxic_patterns)
    )
]

# Print out the results
print(too_broad[["toxic_any", "toxic_elements", "toxic_groups"]].value_counts(dropna=False))

# Grab the ones marked with toxic elements
my_toxic_elements = too_broad[too_broad["toxic_elements"]].copy()

# Print out the results
print("False Positives Elements:")
print(my_toxic_elements["smiles"].values)

# Grab the ones marked with toxic groups
my_toxic_groups = too_broad[too_broad["toxic_groups"]].copy()

# Print out the results
print("False Positives Groups:")
print(my_toxic_groups["smiles"].values)

# Too narrow: See which molecules are not tagged as toxic but ARE toxic (based on the tox21 dataset)
too_narrow = tox21[(tox21["toxic_tag"] == 0) & (tox21["toxic_any"] == 1)].copy()

# Let's check to see if the too_narrow molecules are still flagged as not toxic
too_narrow["toxic_elements"] = too_narrow["molecule"].apply(toxic_elements)
too_narrow["toxic_groups"] = too_narrow["molecule"].apply(toxic_groups)


# Print out the results
print(too_narrow[["toxic_any", "toxic_elements", "toxic_groups"]].value_counts(dropna=False))

# Grab the ones not flagged by either toxic elements or toxic groups (so two false negatives)
my_toxic_elements = too_narrow[~too_narrow["toxic_elements"] & ~too_narrow["toxic_groups"]].copy()

# Print out the results
print(f"False Negatives: {len(my_toxic_elements)}")
print(list(my_toxic_elements["smiles"].values))


# Storage
"""

# This is a list of probably true toxic molecules
probably_true_toxic = {
    "O=C(NC(=O)c1c(F)cccc1F)Nc1cc(Cl)c(OC(F)(F)C(F)C(F)(F)F)cc1Cl",
    "ClC1=C(Cl)C2(Cl)C(CBr)CC1(Cl)C2(Cl)Cl",
    "ClC1=C(Cl)C(Cl)(Cl)C(Cl)=C1Cl",
    "O=C1OC2(c3cc(Br)c(O)c(Br)c3Oc3c2cc(Br)c(O)c3Br)c2c(Cl)c(Cl)c(Cl)c(Cl)c21",
    "COC(CNC(=O)c1ccccc1OCC(=O)O)C[Hg]O",
    "O=C1OC2(c3cc(Br)c([O-])cc3Oc3c2cc(Br)c([O-])c3[Hg]O)c2ccccc21.[Na+].[Na+]",
    "ClC(Cl)(Cl)C(Cl)(Cl)Cl",
    "ClC1C(Cl)C(Cl)C(Cl)C(Cl)C1Cl",
    "O=C(O)C1C(C(=O)O)C2(Cl)C(Cl)=C(Cl)C1(Cl)C2(Cl)Cl",
    "O=P(OC(CCl)CCl)(OC(CCl)CCl)OC(CCl)CCl",
    "Cl[C@H]1[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H]1Cl",
    "CCC(Cl)CCC(Cl)C(Cl)CC(Cl)C(Cl)C(Cl)CCl",
    "ClC1(Cl)C2(Cl)C3(Cl)C4(Cl)C(Cl)(Cl)C5(Cl)C3(Cl)C1(Cl)C5(Cl)C24Cl",
    "ClC(Cl)(Cl)c1ccc(C(Cl)(Cl)Cl)cc1",
    "Brc1cc(Oc2cc(Br)c(Br)c(Br)c2Br)c(Br)c(Br)c1Br",
    "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",
    "Clc1cc(Cl)c(-c2cc(Cl)c(Cl)cc2Cl)cc1Cl",
    "[As]#[In]",
    "O=C(Cl)c1c(Cl)c(Cl)c(C(=O)Cl)c(Cl)c1Cl",
    "O=C(C(Cl)(Cl)Cl)C(Cl)(Cl)Cl",
    "ClC(Cl)=C(Cl)C(Cl)=C(Cl)Cl",
    "Brc1cc(Br)c(Oc2cc(Br)c(Br)cc2Br)cc1Br",
    "O=[Cd]",
    "CC(C)(c1cc(Br)c(OCC(Br)CBr)c(Br)c1)c1cc(Br)c(OCC(Br)CBr)c(Br)c1",
    "FC(F)(Cl)C(F)(Cl)Cl",
    "O=S(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F.[K+]",
    "O=S(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F.[Li+]",
    "FC(Cl)(Cl)C(F)(Cl)Cl",
    "COC(=O)C1=C(C)NC(C)=C(C(=O)OCCN2CCN(C(c3ccccc3)c3ccccc3)CC2)C1c1cccc([N+](=O)[O-])c1.Cl.Cl",
    "Cc1ccc(C(=O)c2cc(O)c(O)c([N+](=O)[O-])c2)cc1",
    "CCCCCCCCCCCCCCCCOP(=O)([O-])OCC[N+](C)(C)C",
    "Cc1nc(-c2ccc(OCC(C)C)c(C#N)c2)sc1C(=O)O",
    "CCCCC(CC)COC(=O)C(C#N)=C(c1ccccc1)c1ccccc1",
    "N#Cc1nn(-c2c(Cl)cc(C(F)(F)F)cc2Cl)c(N)c1S(=O)C(F)(F)F",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "CN(C)CCC[C@@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21.O=C(O)C(=O)O",
    "COP(=O)(OC)C(O)C(Cl)(Cl)Cl",
    "O=[N+]([O-])c1cc([As](=O)(O)O)ccc1O",
    "O=[N+]([O-])c1ccc([As](=O)(O)O)cc1",
    "Cc1c(Cl)c(=O)oc2cc(OP(=O)(OCCCl)OCCCl)ccc12",
    "CO[C@@]1(NC(=O)CSCC#N)C(=O)N2C(C(=O)[O-])=C(CSc3nnnn3C)CS[C@@H]21.[Na+]",
    "CC(=O)Oc1ccccc1C(=O)Nc1ncc([N+](=O)[O-])s1",
    "C[C@@H]1CC(=O)NN=C1c1ccc(NN=C(C#N)C#N)cc1",
    "C[C@@H]1CC(=O)NN=C1c1ccc(NN=C(C#N)C#N)cc1",
    "N#C/C(=C1\\SCC(c2ccccc2Cl)S1)n1ccnc1",
    "C/C1=C2/N=C(/C=C3\\N=C(/C(C)=C4/[C@@H](CCC(N)=O)[C@](C)(CC(N)=O)[C@](C)([C@@H]5N=C1[C@](C)(CCC(=O)NC[C@@H](C)OP(=O)([O-])O[C@H]1[C@@H](O)[C@@H](n6cnc7cc(C)c(C)cc76)O[C@@H]1CO)[C@H]5CC(N)=O)N4[Co+]C#N)[C@@](C)(CC(N)=O)[C@@H]3CCC(N)=O)C(C)(C)[C@@H]2CCC(N)=O",
    "O=C1CN(N=Cc2ccc(-c3ccc([N+](=O)[O-])cc3)o2)C(=O)[N-]1.[Na+]",
    "CS(=O)(=O)Nc1ccc([N+](=O)[O-])cc1Oc1ccccc1",
    "Cc1c(C(N)=O)cc([N+](=O)[O-])cc1[N+](=O)[O-]",
    "Cc1ncc([N+](=O)[O-])n1CC(O)CCl",
    "Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1",
    "Cc1ncc([N+](=O)[O-])n1C",
    "NC(=O)c1ccc([N+](=O)[O-])cc1Cl",
    "O=C(NCCO[N+](=O)[O-])c1cccnc1",
    "O=[N+]([O-])/N=C1\\NCCN1Cc1ccc(Cl)nc1",
}

# Filter out the probably true toxic molecules
too_broad = too_broad[~too_broad["smiles"].isin(probably_true_toxic)]
"""