from collections import defaultdict
from io import StringIO
import os
#from itertools import combinations
from pkg_resources import resource_filename
from urllib.request import urlopen

import numpy as np

import simtk.openmm.app as app
import simtk.openmm as openmm
import simtk.unit as unit

from openforcefield.topology import Molecule
from openmmtools.constants import kB
from perses.utils.smallmolecules import render_atom_mapping, smiles_to_mol, sdf_to_mols
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
from perses.rjmc import atom_mapper


def test_ring_breaking_detection():
    """
    Test the detection of ring-breaking transformations.

    """
    molecule1 = smiles_to_mol("c1ccc2ccccc2c1", "naphthalene", 1)
    molecule2 = smiles_to_mol("c1ccccc1", "benzene", 1)

    # Allow ring breaking
    new_to_old_atom_map = atom_mapper._get_mol_atom_map(molecule1, molecule2, allow_ring_breaking=True)
    if not len(new_to_old_atom_map) > 0:
        filename = 'mapping-error.png'
        #render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
        msg = 'Napthalene -> benzene transformation with allow_ring_breaking=True is not returning a valid mapping\n'
        msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
        msg += str(new_to_old_atom_map)
        raise Exception(msg)

    new_to_old_atom_map = atom_mapper._get_mol_atom_map(molecule1, molecule2, allow_ring_breaking=False)
    if new_to_old_atom_map is not None: # atom mapper should not retain _any_ atoms in default mode
        filename = 'mapping-error.png'
        #render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
        msg = 'Napthalene -> benzene transformation with allow_ring_breaking=False is erroneously allowing ring breaking\n'
        msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
        msg += str(new_to_old_atom_map)
        raise Exception(msg)


def _load_JACS(dataset_name):
    dataset_path = f'data/schrodinger-jacs-datasets/{dataset_name}_ligands.sdf'
    sdf_filename = resource_filename('perses', dataset_path)
    return sdf_to_mols(sdf_filename)


def test_molecular_atom_mapping():
    """
    Test the creation of atom maps between pairs of molecules from the JACS benchmark set.

    """

    # Test mappings for JACS dataset ligands
    for dataset_name in ['CDK2']: #, 'p38', 'Tyk2', 'Thrombin', 'PTP1B', 'MCL1', 'Jnk1', 'Bace']:
        # Read molecules
        molecules = _load_JACS(dataset_name)

        # Build atom map for some transformations.
        #for (molecule1, molecule2) in combinations(molecules, 2): # too slow
        molecule1 = molecules[0]
        for i, molecule2 in enumerate(molecules[1:]):
            new_to_old_atom_map = atom_mapper._get_mol_atom_map(molecule1, molecule2)
            # Make sure we aren't mapping hydrogens onto anything else
            #atoms1 = [atom for atom in molecule1.GetAtoms()]
            #atoms2 = [atom for atom in molecule2.GetAtoms()]
            #for (index2, index1) in new_to_old_atom_map.items():
            #    atom1, atom2 = atoms1[index1], atoms2[index2]
            #    if (atom1.GetAtomicNum()==1) != (atom2.GetAtomicNum()==1):
            filename = 'mapping-error-%d.png' % i
            render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
            #msg = 'Atom atomic number %d is being mapped to atomic number %d\n'
            msg = f'molecule 1 : {molecule1.to_smiles()}\n'
            msg += f'molecule 2 : {molecule2.to_smiles()}\n'
            msg += f'Wrote atom mapping to {filename} for inspection; please check this.'
            msg += str(new_to_old_atom_map)
            print(msg)
            #        raise Exception(msg)
            # TODO: this is not a real test


def test_map_strategy():
    """
    Test the creation of atom maps between pairs of molecules from the JACS benchmark set.

    """
    # Test mappings for JACS dataset ligands
    for dataset_name in ['Jnk1']:
        molecules = _load_JACS(dataset_name)
        atom_expr = None # oechem.OEExprOpts_IntType
        bond_expr = None # oechem.OEExprOpts_RingMember

        # the 0th and 1st Jnk1 ligand have meta substituents that face opposite eachother
        # in the active site. Using `map_strategy=matching_criterion` should align these groups, and put them
        # both in the core. Using `map_strategy=geometry` should see that the orientations differ and chose
        # to unmap (i.e. put both these groups in core) such as to get the geometry right at the expense of
        # mapping fewer atoms
        new_to_old_atom_map = atom_mapper._get_mol_atom_map(molecules[0], molecules[1],atom_expr=atom_expr,bond_expr=bond_expr)
        assert len(new_to_old_atom_map) == 37, 'Expected meta groups methyl C to map onto ethyl O'

        new_to_old_atom_map = atom_mapper._get_mol_atom_map(molecules[0], molecules[1],atom_expr=atom_expr,bond_expr=bond_expr,map_strategy='geometry')
        assert len(new_to_old_atom_map) == 35,  'Expected meta groups methyl C to NOT map onto ethyl O as they are distal in cartesian space'

# TODO: move to pytest so these can be parameterized
def test_simple_heterocycle_mapping(smiles_pairs = [('c1ccccc1', 'c1ncccc1')]):
    """
    Test the ability to map conjugated heterocycles (that preserves all rings).  Will assert that the number of ring members in both molecules is the same.
    """
    def _atom_by_index(mol, idx):
        for atom in mol.atoms:
            if atom.molecule_atom_index == idx:
                return atom
        else:
            raise Exception(f'{idx} not found in {mol}')
    # TODO: generalize this to test for ring breakage and closure.
    for smiles_old, smiles_new in smiles_pairs:

        old_mol, new_mol = smiles_to_mol(smiles_old), smiles_to_mol(smiles_new)
        #raise Exception(f'{smiles_old} {old_mol} --> {smiles_new} {new_mol}')
        new_to_old_map = atom_mapper._get_mol_atom_map(
            old_mol, new_mol, allow_ring_breaking=False)
        #raise Exception(f'{new_to_old_map}')
        #assert that the number of ring members is consistent in the mapping...
        num_hetero_maps = 0
        for new_index, old_index in new_to_old_map.items():
            old_atom = _atom_by_index(old_mol, old_index)
            new_atom = _atom_by_index(new_mol, new_index)

            if old_atom.is_in_ring and new_atom.is_in_ring:
                if old_atom.atomic_number != new_atom.atomic_number:
                    num_hetero_maps += 1

        assert num_hetero_maps > 0, f"there are no differences in atomic number mappings in {smiles_old}, {smiles_new}"
