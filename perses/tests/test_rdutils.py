"""
Test RDkit module
"""
from collections import defaultdict
import pytest

import numpy as np
from scipy.spatial.distance import cdist

from rdkit import Chem
from rdkit.Chem import AllChem
from openforcefield.topology import Molecule

from perses.utils import rdutils, smallmolecules
# TODO: MappingExpression could be in a utility class?
from perses.rjmc.atom_mapper import MappingExpression
from .sample_data import load_JACS


@pytest.mark.parametrize(
    "smiles,expected_core_smiles",
    [("COc1ccccc1", "c1ccccc1"),
     ("COc1ccccc1-C-c2ccccc2", "c1ccccc1-C-c2ccccc2"),
    ])
def test_get_scaffold(smiles, expected_core_smiles):
    mol = smallmolecules.smiles_to_mol(smiles)
    core = rdutils.get_scaffold(mol)
    expected_core = smallmolecules.smiles_to_mol(expected_core_smiles)
    cleaned_expected_core_smiles = expected_core.to_smiles(
        explicit_hydrogens=False)
    core_smiles = core.to_smiles(explicit_hydrogens=False)
    assert cleaned_expected_core_smiles == core_smiles


@pytest.mark.parametrize(
    "smiles,expected_atom_fp,expected_bond_fp",
    [("c1ccccc1", [1 << 4]*6, [1 << 4]*6),
     ("c1ccccc1(CC[N@@]2CNNC2)",
      [1 << 4]*6 + [0, 0] + [1 << 5]*5,
      [1 << 4]*6 + [0, 0, 0] + [1 << 5]*5),
     ("OC1[C@@]2[C@]1CC2", [0, 1 << 7] + [0b11 << 6]*2 + [1 << 6]*2,
      [0, 1 << 7, 1 << 7, 0b11 << 6] + [1 << 6] * 3)
    ])
def test_ring_fingerprints(smiles, expected_atom_fp, expected_bond_fp):
    mol = smallmolecules.smiles_to_mol(smiles)
    atom_fp, bond_fp = rdutils.ring_fingerprints(mol)
    # ignore hydrogens
    num_hydrogens = mol.n_atoms - len(expected_atom_fp)
    extra_zeros = [0]*num_hydrogens
    assert atom_fp == expected_atom_fp + extra_zeros
    # Can't predict order of bonds, so just make sure the right number of
    # atoms are assigned to the right size of rings
    assert sorted(bond_fp) == sorted(expected_bond_fp + extra_zeros)


@pytest.fixture
def jnk_compounds():
    return load_JACS("Jnk1")

# There appears to be a bug in RDKit where passing CompareAnyHeavyAtom to MCS
# prevents it from finding the second ring in the Jnk compounds
# when bondCompare is not CompareOrderExact???
# Also, we get the ring backwards since RDKit MCS tries to match the
# flipped ring and doesn't give us an MCS with the original ring

@pytest.mark.parametrize(
    "name1,name2,expected_drop_heavies1,expected_drop_heavies2,expected_matches",
    [("18629-1", "18634-1", {19}, {18, 19, 24, 25}, 1),
     ("18660-1", "18634-1", {0, 1, 2, 3, 4, 29}, {0, 25}, 1),
     ("18625-1", "18659-1", {11, 18}, {0, 1, 20, 21, 26, 27}, 1),
    ])
def test_mcs_jnk(name1, name2,
                 expected_drop_heavies1, expected_drop_heavies2,
                 expected_matches,
                 jnk_compounds):
    mol1 = jnk_compounds[name1]
    mol2 = jnk_compounds[name2]
    assert (mol1.name, mol2.name) == (name1, name2)
    #mapping_expr = MappingExpression('match_aromaticity', 'match_ring')
    heavies1 = {atom.molecule_atom_index for atom in mol1.atoms
                if atom.atomic_number != 1}
    heavies2 = {atom.molecule_atom_index for atom in mol2.atoms
                if atom.atomic_number != 1}
    matches = rdutils.mcs3d(mol1, mol2)
    assert len(matches) == expected_matches

    def _heavies_message(mol1, mol2, dropped_atoms):
        coords1 = mol1.conformers[0]
        coords2 = mol2.conformers[0]
        distances = cdist(coords1, coords2, 'euclidean')
        details = ""
        for idx in dropped_atoms:
            closest = np.amin(distances[idx])
            atom = mol1.atoms[idx]
            atomic_num = atom.atomic_number
            bonded = defaultdict(int)
            for other_atom in atom.bonded_atoms:
                bonded[other_atom.atomic_number] += 1
            bonded_desc = {k: v for k, v in bonded.items()}
            details += f"\n{idx}: #{atomic_num} {closest} - {bonded_desc}"
        return details

    for match in matches:

        drop_heavies1 = heavies1 - {mol1.atoms[v].molecule_atom_index
                                    for v in match.values()}
        drop_heavies2 = heavies2 - {mol2.atoms[k].molecule_atom_index
                                    for k in match.keys()}
        import ipdb;ipdb.set_trace()
        assert drop_heavies1 == expected_drop_heavies1, \
            _heavies_message(mol1, mol2, drop_heavies1)
        assert drop_heavies2 == expected_drop_heavies2, \
            _heavies_message(mol2, mol2, drop_heavies2)



@pytest.mark.parametrize(
    "smiles1,smiles2,scaffold_smiles,expected_num_matches,expected_atomic_nums",
    [('C1CCOC[C@]1(NC)O', 'C1CCOC[C@@]1(NC)O','C1CCOCC1',
      6, {6: 5, 8: 1, 1: 8}),
     ('c1cc(O)cnc1-[C@@](CCO)-c2cnccc2', 'c1cc(O)cnc1-[C@H]-c2cnccc2',
      'c1cccnc1-C-c2cnccc2', 1, {6: 11, 8: 1, 7: 2, 1: 8})
    ])
def test_mcs3d(smiles1, smiles2, scaffold_smiles,
               expected_num_matches, expected_atomic_nums):
    seed = 0xf00a
    m1 = Chem.MolFromSmiles(smiles1)
    m2 = Chem.MolFromSmiles(smiles2)
    scaffold = Chem.AddHs(Chem.MolFromSmiles(scaffold_smiles))
    AllChem.EmbedMolecule(scaffold, randomSeed=seed, useBasicKnowledge=True)
    scaff_noh = Chem.RemoveHs(scaffold)
    scaff_conf = scaff_noh.GetConformer()
    def _embed_add_h(m):
        match_m = m.GetSubstructMatch(scaff_noh)
        m_map = {}
        for i,idx in enumerate(match_m):
            m_map[idx]=scaff_conf.GetAtomPosition(i)
            m_h = Chem.AddHs(m)
            # generating conformations that match core atoms works best using random coordinates:
            AllChem.EmbedMolecule(m_h, randomSeed=seed,
                                  coordMap=m_map, useRandomCoords=True)
        return m_h
    m1_h = _embed_add_h(m1)
    m2_h = _embed_add_h(m2)
    mol1 = Molecule.from_rdkit(m1_h)
    mol2 = Molecule.from_rdkit(m2_h)
    matches = rdutils.mcs3d(mol1, mol2)
    #match = sorted(matches, key=lambda x: len(x), reverse=True)[0]
    assert len(matches) == expected_num_matches
    match = matches[0]
    match_atomic_nums = defaultdict(int)
    for m1_idx in match.values():  # Note that keys of match are mol2 indexes
        match_atomic_nums[mol1.atoms[m1_idx].atomic_number] += 1
    assert match_atomic_nums == expected_atomic_nums
