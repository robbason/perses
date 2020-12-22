"""
Functions that should be modularized so choice of openeye / rdkit
is abstracted away. Use rdkit mol internally, but take as inputs
openff Molecule objects.
"""
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFMCS import FindMCS, AtomCompare, BondCompare
from rdkit.Chem import MolFromSmarts
from openforcefield.topology import Molecule


RDKIT_MCS_ATOM = {
    'match_aromaticity': AtomCompare.CompareAnyHeavyAtom,
    'match_hybridization': AtomCompare.CompareAnyHeavyAtom,
    'match_hybrid_and_atom_type': AtomCompare.CompareElements
}

RDKIT_MCS_BOND = {
    'match_ring': BondCompare.CompareOrderExact,
    'match_aromaticity': BondCompare.CompareOrderExact,
    'match_hybridization': BondCompare.CompareOrderExact,
    'match_hybrid_and_atom_type': BondCompare.CompareOrderExact
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


def mcs(mol1, mol2, mapping_expr):
    """
    See https://www.rdkit.org/docs/source/rdkit.Chem.rdFMCS.html
    molA, molB openff Molecule
    mapping_expr MappingExpression namedtuple from atom_mapper
    """
    m1 = mol1.to_rdkit()
    m2 = mol2.to_rdkit()
    kwargs = _mcs_kwargs(mapping_expr)
    res = FindMCS([m1, m2], timeout=MCS_MAX_SECONDS, **kwargs)
    if res.canceled:
        raise Exception(f"Timeout on MCS of {mol1} and {mol2}")
    matches1 = m1.GetSubstructMatches(res.queryMol)
    matches2 = m2.GetSubstructMatches(res.queryMol)
    #return res, matches1, matches2
    # new to old map
    return [dict(zip(match2, match1))
            for match1 in matches1
            for match2 in matches2]
