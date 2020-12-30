"""
Functions that should be modularized so choice of openeye / rdkit
is abstracted away. Use rdkit mol internally, but take as inputs
openff Molecule objects.
"""
import numpy as np
from scipy.spatial.distance import cdist

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS
from rdkit.Chem import MolFromSmarts
from rdkit import Chem
from openforcefield.topology import Molecule


RDKIT_MCS_ATOM = {
    'match_aromaticity': rdFMCS.AtomCompare.CompareAnyHeavyAtom,
    'match_hybridization': rdFMCS.AtomCompare.CompareAnyHeavyAtom,
    'match_hybrid_and_atom_type': rdFMCS.AtomCompare.CompareElements
}

RDKIT_MCS_BOND = {
    'match_ring': rdFMCS.BondCompare.CompareOrderExact,
    'match_aromaticity': rdFMCS.BondCompare.CompareOrderExact,
    'match_hybridization': rdFMCS.BondCompare.CompareOrderExact,
    'match_hybrid_and_atom_type': rdFMCS.BondCompare.CompareOrderExact
}

MCS_MAX_SECONDS=100


def get_scaffold(mol):
    m1 = mol.to_rdkit()
    core = MurckoScaffold.GetScaffoldForMol(m1)
    return Molecule.from_rdkit(core)


def _mcs_kwargs(mapping_expr):
    kwargs = {
        'atomCompare': RDKIT_MCS_ATOM[mapping_expr.atom],
        'bondCompare': RDKIT_MCS_BOND[mapping_expr.bond],
        'matchValences': mapping_expr.atom != 'match_aromaticity',
        'ringMatchesRingOnly': mapping_expr.bond != 'match_ring', # TODO
        # completeRingsOnly: TBD
        # matchChiralTag
        # matchFormalCharge - should be False (default)
    }
    return kwargs


def ring_fingerprints(mol, max_ring_size=10):
    """
    Returns (for each of atoms and bonds)
    a list of integers with each bit set based on whether the atom or bond
    is in a ring of that size. Highest order bit is for ring size 3.
    e.g. (in binary): 100...0 means it's only in a 3-member ring
    101100... means it's in a 3-member ring, a 5-member ring,
    and a 6-member ring
    """
    MIN_RING_SIZE = 3
    def _fingerprint(ring_method, idx, max_ring_size):
        fingerprint = 0
        for ring_size in range(MIN_RING_SIZE, max_ring_size + 1):
            fingerprint <<= 1
            if ring_method(idx, ring_size):
                fingerprint |= 1
        return fingerprint

    def _atom_ring_fingerprint(ring_info, atom_idx, max_ring_size):
        return _fingerprint(
            ring_info.IsAtomInRingOfSize, atom_idx, max_ring_size)

    def _bond_ring_fingerprint(ring_info, bond_idx, max_ring_size):
        return _fingerprint(
            ring_info.IsBondInRingOfSize, bond_idx, max_ring_size)

    rdmol = mol.to_rdkit()
    ring_info = rdmol.GetRingInfo()

    atom_ring_fingerprints = [
        _atom_ring_fingerprint(ring_info, atom_idx, max_ring_size)
        for atom_idx in range(rdmol.GetNumAtoms())]
    bond_ring_fingerprints = [
        _bond_ring_fingerprint(ring_info, bond_idx, max_ring_size)
        for bond_idx in range(rdmol.GetNumBonds())]
    return atom_ring_fingerprints, bond_ring_fingerprints


def _set_atom_numbers(rdmol, nums):
    """
    Set numeric attribute on atoms for use in MCS
    """
    for atom, num in zip(rdmol.GetAtoms(), nums):
        atom.SetAtomMapNum(num)


"""class BondCustomNumCompare(BondCompare.BondCompareUserData):
    def __call__(self, p, mol1, bond1, mol2, bond2):
        b1 = mol1.GetBondWithIdx(bond1)
        b2 = mol2.GetBondWithIdx(bond2)
   """

def mcs(mol1, mol2, mapping_expr, atom_nums1=None, atom_nums2=None):
    """
    See https://www.rdkit.org/docs/source/rdkit.Chem.rdFMCS.html
    molA, molB openff Molecule
    mapping_expr MappingExpression namedtuple from atom_mapper
    """
    m1 = mol1.to_rdkit()
    m2 = mol2.to_rdkit()

    if atom_nums1 and atom_nums2:
        _set_atom_numbers(m1, atom_nums1)
        _set_atom_numbers(m2, atom_nums2)

    if (atom_nums1 and not atom_nums2) or (atom_nums2 and not atom_nums1):
        raise ValueError("Must set atom nums on both mols or none")
    kwargs = _mcs_kwargs(mapping_expr)
    res = rdFMCS.FindMCS([m1, m2], timeout=MCS_MAX_SECONDS, **kwargs)
    if res.canceled:
        raise Exception(f"Timeout on MCS of {mol1} and {mol2}")
    matches1 = m1.GetSubstructMatches(res.queryMol,
                                      uniquify=False, useChirality=True)
    matches2 = m2.GetSubstructMatches(res.queryMol,
                                      uniquify=False, useChirality=True)
    #return res, matches1, matches2
    # new to old map
    # Remove duplicates
    raw_matches = {tuple(sorted(list(zip(match2, match1))))
            for match1 in matches1
            for match2 in matches2}
    return [dict(mp) for mp in raw_matches]


def _atom_type_map(mol1, mol2):
    """
    Returns a numpy array of booleans, where each
    position tells us whether that atom pair is either both H or both not H
    """
    mol1_h = np.array([(atom.atomic_number == 1) for atom in mol1.atoms],
                      dtype=bool)
    mol2_h = np.array([(atom.atomic_number == 1) for atom in mol2.atoms],
                      dtype=bool)
    # Need array to be of shape len(mol1), len(mol2)
    # TODO: make sure this is right
    h_mesh2, h_mesh1 = np.meshgrid(mol2_h, mol1_h)
    h_map = h_mesh1 * h_mesh2
    noth_map = ~h_mesh1 * ~h_mesh2
    return h_map | noth_map


def _atom_type_map_exact(mol1, mol2):
    """
    Require atomic number exact match

    """
    m1_nums = np.array([atom.atomic_number for atom in mol1.atoms])
    m2_nums = np.array([atom.atomic_number for atom in mol2.atoms])
    msh2, msh1 = np.meshgrid(m2_nums, m1_nums)
    return msh1 == msh2


def _distance_map(mol1, mol2, tolerance):
    """
    Returns a numpy array of booleans, where each
    position tells us whether that atom pair is either both H or both not H
    and whether the atom pair is within tolerance
    """
    coords1 = mol1.conformers[0]
    coords2 = mol2.conformers[0]
    all_to_all = cdist(coords1, coords2, 'euclidean')
    return all_to_all <= tolerance


def mcs3d(mol1, mol2, tolerance=1.0):
    """
    Ignore any atom for MCS that isn't within tolerance angstroms
    from a matching atom - H matches only H, heavy atoms match
    This version uses rdkit with 3D MCS, only in my fork now (12/2020)
    """
    m1 = mol1.to_rdkit()
    m2 = mol2.to_rdkit()
    res = rdFMCS.FindMCS(
        [m1, m2],
        atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        maxDistance=tolerance,
        timeout=MCS_MAX_SECONDS,)
    if res.canceled:
        raise Exception(f"Timeout on MCS of {mol1} and {mol2}")


def mcs3d_old(mol1, mol2, tolerance=1.5):
    # TODO: tolerance was 0.3 in perses code. But Jnk test fails, clearly so??
    """
    Ignore any atom for MCS that isn't within tolerance angstroms
    from a matching atom - H matches only H, heavy atoms match
    This version removes atoms from the match based on geometry
    """
    m1 = mol1.to_rdkit()
    m2 = mol2.to_rdkit()
    res = rdFMCS.FindMCS(
        [m1, m2],
        atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        timeout=MCS_MAX_SECONDS,)
    if res.canceled:
        raise Exception(f"Timeout on MCS of {mol1} and {mol2}")
    #import ipdb;ipdb.set_trace()
    # TODO: RDKit MCS is not geometry aware, and in too many cases
    # this is a problem. e.g. if there's a ring that could match both ways,
    # but a substituent points off one way in one mol and the other in the other,
    # mcs will only get the SMARTS with the substituents mapped to each other.
    # post-filtering on geometry won't save us!
    matches1 = m1.GetSubstructMatches(
        res.queryMol, uniquify=False, useChirality=True)
    matches2 = m2.GetSubstructMatches(
        res.queryMol, uniquify=False, useChirality=True)

    # Since the Nth element of match1 corresponds to the
    # same query atom as the Nth element of match2, we can zip the lists
    # together to find correspondence between match1 and match2.
    raw_maps = {tuple(sorted(list(zip(match2, match1))))
            for match1 in matches1
            for match2 in matches2}

    distance_map = _distance_map(mol1, mol2, tolerance)
    atom_type_map = _atom_type_map(mol1, mol2)
    inclusion_map = distance_map & atom_type_map

    def _filter_geom(raw_map):
        return tuple([(idx2, idx1) for idx2, idx1 in raw_map
                if inclusion_map[idx1][idx2]])
    #import ipdb;ipdb.set_trace()
    filtered_maps = set([_filter_geom(raw_map) for raw_map in raw_maps])
    max_match_len = max([len(fm) for fm in filtered_maps])
    longest_filtered = [dict(fm) for fm in filtered_maps
                        if len(fm) == max_match_len]
    return longest_filtered


def mcs3d_pregeom(mol1, mol2, tolerance=0.3):
    """
    Ignore any atom for MCS that isn't within tolerance angstroms
    from a matching atom - H matches only H, heavy atoms match
    """
    distance_map = _distance_map(mol1, mol2, tolerance)
    atom_type_map = _atom_type_map(mol1, mol2)
    inclusion_map = distance_map & atom_type_map
    keep1 = np.amax(inclusion_map, axis=1)
    keep2 = np.amax(inclusion_map, axis=0)
    def _keep_atoms(mol, keep_list):
        m = Chem.RWMol(mol.to_rdkit())
        for i in range(m.GetNumAtoms(),0,-1):
            idx = i-1
            if not keep_list[idx]:
                m.RemoveAtom(idx)
        return m

    reduced_m1 = _keep_atoms(mol1, keep1)
    reduced_m2 = _keep_atoms(mol2, keep2)
    res = rdFMCS.FindMCS(
        [reduced_m1, reduced_m2],
        atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        timeout=MCS_MAX_SECONDS,)
    if res.canceled:
        raise Exception(f"Timeout on MCS of {mol1} and {mol2}")
    orig_m1 = mol1.to_rdkit()
    orig_m2 = mol2.to_rdkit()
    matches1 = orig_m1.GetSubstructMatches(
        res.queryMol, uniquify=False, useChirality=True)
    matches2 = orig_m2.GetSubstructMatches(
        res.queryMol, uniquify=False, useChirality=True)
    raw_matches = {tuple(sorted(list(zip(match2, match1))))
            for match1 in matches1
            for match2 in matches2}
    return [dict(mp) for mp in raw_matches]


def visualize_mapping(mapping):
    from rdkit.Chem.Draw import IPythonConsole # Needed to show molecules
    from rdkit.Chem import Draw
    rA = mapping.current_mol.to_rdkit()
    rB = mapping.proposed_mol.to_rdkit()
    def mol_with_atom_index(mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        return mol
    mol_with_atom_index(rA)
    mol_with_atom_index(rB)
    listB = mapping.atom_map.keys()
    listA = mapping.atom_map.values()
    return Draw.MolsToGridImage([rA,rB], highlightAtomLists=[listA, listB],
                                subImgSize=(400, 400))
