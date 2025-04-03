import random
import pandas as pd
from rdkit import Chem


def detect_double_bond_stereochemistry(df, verbose=True):
    """
    Analyze a dataframe of molecules to check for E/Z stereochemistry at double bonds.

    Args:
        df (pd.DataFrame): DataFrame containing RDKit molecules in 'molecule' column
        verbose (bool): Whether to print detailed information about molecules with E/Z stereochemistry

    Returns:
        dict: Statistics about E/Z stereochemistry in the dataset
    """

    # Sample molecules for detailed analysis
    sample_size = min(100, len(df))
    sampled_indices = random.sample(range(len(df)), sample_size)

    # Count statistics
    total_molecules = len(df)
    molecules_with_ez = 0
    ez_bonds_total = 0

    # For detailed reporting
    molecules_with_ez_details = []

    # Method 1: Using FindPotentialStereo (modern approach)
    for idx in sampled_indices:
        mol = df.iloc[idx]["molecule"]
        if mol is None:
            continue

        # Ensure stereochemistry is assigned
        Chem.AssignStereochemistry(mol, force=True)

        # Find stereo information including bond stereo
        stereo_info = Chem.FindPotentialStereo(mol)

        ez_bonds = []
        for element in stereo_info:
            if element.type == Chem.StereoType.Bond_Double and element.specified == Chem.StereoSpecified.Specified:
                if element.descriptor in [Chem.StereoDescriptor.Bond_Cis, Chem.StereoDescriptor.Bond_Trans]:
                    bond = mol.GetBondWithIdx(element.centeredOn)
                    begin_atom = bond.GetBeginAtom().GetSymbol()
                    end_atom = bond.GetEndAtom().GetSymbol()
                    stereo_desc = "Cis (Z)" if element.descriptor == Chem.StereoDescriptor.Bond_Cis else "Trans (E)"

                    ez_bonds.append(
                        {"bond_idx": element.centeredOn, "atoms": f"{begin_atom}={end_atom}", "stereo": stereo_desc}
                    )

        if ez_bonds:
            molecules_with_ez += 1
            ez_bonds_total += len(ez_bonds)

            if verbose:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                molecules_with_ez_details.append({"idx": idx, "smiles": smiles, "ez_bonds": ez_bonds})

    # Check the entire dataset with a faster method
    total_with_ez = 0
    for mol in df["molecule"]:
        if mol is None:
            continue

        # Check for double bond stereochemistry in the SMILES
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if "/" in smiles or "\\" in smiles:  # Quick check for potential E/Z notation
            total_with_ez += 1

    # Prepare results
    results = {
        "total_molecules": total_molecules,
        "molecules_with_ez_sampled": molecules_with_ez,
        "ez_bonds_total_sampled": ez_bonds_total,
        "estimated_percent_with_ez": (molecules_with_ez / sample_size) * 100 if sample_size > 0 else 0,
        "total_with_potential_ez": total_with_ez,
        "percent_with_potential_ez": (total_with_ez / total_molecules) * 100 if total_molecules > 0 else 0,
    }

    # Print detailed results if requested
    if verbose and molecules_with_ez_details:
        print(f"Found {molecules_with_ez} molecules with E/Z stereochemistry in sample of {sample_size}")
        print(f"Estimated {results['estimated_percent_with_ez']:.2f}% of molecules have E/Z stereochemistry")
        print("\nDetailed examples:")

        for idx, mol_info in enumerate(molecules_with_ez_details[:5]):  # Show first 5 examples
            print(f"\nExample {idx + 1}:")
            print(f"SMILES: {mol_info['smiles']}")
            print("E/Z bonds:")
            for bond in mol_info["ez_bonds"]:
                print(f"  - Bond {bond['bond_idx']} ({bond['atoms']}): {bond['stereo']}")

    return results


if __name__ == "__main__":

    # Set pandas display options
    pd.options.display.max_columns = 20
    pd.options.display.max_colwidth = 200
    pd.options.display.width = 1400

    # Test data
    # Create test molecules with known E/Z stereochemistry
    test_smiles = [
        # E (trans) examples
        "C/C=C/C",  # trans-2-butene
        "C/C=C/Cl",  # trans-2-chloro-2-butene
        "ClC=CCl",  # non-stereo notation
        "Cl/C=C/Cl",  # trans-1,2-dichloroethene
        # Z (cis) examples
        "C/C=C\\C",  # cis-2-butene
        "C/C=C\\Cl",  # cis-2-chloro-2-butene
        "Cl/C=C\\Cl",  # cis-1,2-dichloroethene
        # More complex examples
        "C/C=C/C=C",  # trans-2,4-hexadiene
        "C/C=C\\C=C",  # mix of cis and trans
        "C/C=C/C=C/C",  # all-trans-2,4,6-octatriene
        "C/C(Cl)=C\\C",  # substituted example
        # Non-stereochemical double bonds
        "C=C",  # ethene (no stereochemistry)
        "C=CC=C",  # 1,3-butadiene (no specified stereochemistry)
        "C1=CCCCC1",  # cyclohexene (no stereochemistry possible)
        # Compare with chiral centers
        "C[C@H](Cl)Br",  # chiral molecule
        "CC(Cl)Br"  # non-chiral notation
        "N[C@H]1CC[C@@H](CC1)[NH2+]CCF",  # From RDKIT/Github discussion example
    ]

    # AQSol Smiles
    aqsol_smiles = [
        r"CCCCCCCC\\C=C\\CCCCCCCCNCCCNCCCNCCCN",
        r"COC1=CC=C(C=C1N\\N=C1/C(=O)C(=CC2=CC=CC=C12)C(=O)NC1=CC(Cl)=CC=C1C)C(=O)NC1=CC=CC=C1",
        r"NC(=O)N\\N=C\\C(O)C(O)C(O)CO",
        r"C1=CC=C(C=C1)\\N=N\\C1=CC=CC=C1",
        r"CC(=O)N\\N=C\\C1=CC=C(O1)[N+]([O-])=O",
        r"CC(=O)OCCN(CCC#N)C1=CC=C(C=C1)\\N=N\\C1=CC=C(C=C1)[N+]([O-])=O",
        r"ClC1=CC=C(Cl)C(N\\N=C2/C(=O)C(=CC3=CC=CC=C23)C(=O)NC2=CC=C3NC(=O)NC3=C2)=C1",
        r"NC1=CC=C(C=C1)\\N=N\\C1=CC=CC=C1",
        r"OC(=O)\\C=C/C=C\\C(O)=O",
        r"CCOC(=O)\\C=C\\C1=CC=CC=C1",
        r"CC(=O)\\C=C\\C1=C(C)CCCC1(C)C",
        r"C\\C(=C/C(O)=O)C(O)=O",
        r"CCC\\C=C\\C",
        r"CC1=NN(C(=O)\\C1=N\\NC1=CC=C(C=C1Cl)C1=CC=C(N\\N=C2/C(C)=NN(C2=O)C2=CC=CC=C2)C(Cl)=C1)C1=CC=CC=C1",
        r"OC(C1=CC2C3C(C1\\C2=C(\\C1=CC=CC=C1)C1=CC=CC=N1)C(=O)NC3=O)(C1=CC=CC=C1)C1=CC=CC=N1",
        r"COC1=CC=C(\\C=C\\C(=O)C2=C(O)C=CC=C2)C=C1",
        r"CC\\C(=C(\\CC)C1=CC=C(O)C=C1)C1=CC=C(O)C=C1",
        r"C\\C=C\\OC1CCC(CC1)O\\C=C\\C",
        r"CC(C)=C[C@@H]1[C@@H](C(=O)O[C@H]2CC(=O)C(C\\C=C/C=C)=C2C)C1(C)C",
        r"CC\\C=C\\C",
        r"COC(=O)C(\\C)=C\\[C@@H]1[C@@H](C(=O)O[C@H]2CC(=O)C(C\\C=C/C=C)=C2C)C1(C)C",
        r"CC1=C(F)C(F)=C(COC(=O)C2C(\\C=C(/Cl)C(F)(F)F)C2(C)C)C(F)=C1F",
        r"CCC(=O)OC\\C=C(/C)\\C=C\\C=C(/C)\\C=C\\C1=C(C)CCCC1(C)C",
        r"CC(=O)C(\\C)=C/C1C(C)=CCCC1(C)C",
        r"CC(=O)C(\\N=N\\C1=CC=CC=C1C(O)=O)C(=O)NC1=CC=C2NC(=O)NC2=C1",
        r"O\\N=C1\\CCCC=C1",
        r"CCCCCCCCCCCCCCCC(=O)NCCCCCCCC\\C=C/CCCCCCCC",
        r"ClC\\C=C/CCl",
        r"CC(=O)C(\\N=N\\C1=CC=C(Cl)C=C1[N+]([O-])=O)C(=O)NC1=CC=C2NC(=O)NC2=C1",
        r"OC(=O)\\C=C(/Cl)C1=CC=CC=C1",
        r"CC(=O)C(\\N=N\\C1=CC=C(C=C1)[N+]([O-])=O)C(=O)NC1=CC=C2NC(=O)NC2=C1",
        r"CC\\C=C/CCO",
    ]
    all_smiles = test_smiles + aqsol_smiles

    # Create molecules
    mols = [Chem.MolFromSmiles(s) for s in all_smiles]

    # Create test dataframe
    df = pd.DataFrame({"smiles": all_smiles, "molecule": mols})

    # Run the detection function
    results = detect_double_bond_stereochemistry(df, verbose=True)
    print("\nOverall Results:")
    print(f"Total molecules: {results['total_molecules']}")
    print(f"Molecules with E/Z stereochemistry in sample: {results['molecules_with_ez_sampled']}")
    print(f"Estimated percent with E/Z stereochemistry: {results['estimated_percent_with_ez']:.2f}%")
    print(f"Total with potential E/Z stereochemistry: {results['total_with_potential_ez']}")
    print(f"Percent with potential E/Z stereochemistry: {results['percent_with_potential_ez']:.2f}%")
