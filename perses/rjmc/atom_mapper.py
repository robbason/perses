import copy
import logging
from collections import namedtuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from perses.utils.openff import get_scaffold
from perses.utils.rdkit import mcs
from perses.utils.smallmolecules import render_atom_mapping


_logger = logging.getLogger(__name__)

# possible values for map_strategy
STRATEGY_CORE = 'core'
STRATEGY_GEOMETRY = 'geometry'
STRATEGY_MATCHING_CRITERION = 'matching_criterion'
STRATEGY_RANDOM = 'random'
STRATEGY_WEIGHTED_RANDOM = 'weighted-random'
STRATEGY_RETURN_ALL = 'return-all'
MAP_STRATEGIES = [
    STRATEGY_CORE, STRATEGY_GEOMETRY, STRATEGY_MATCHING_CRITERION,
    STRATEGY_RANDOM, STRATEGY_WEIGHTED_RANDOM, STRATEGY_RETURN_ALL]

MATCHING_INDEX = 'index'
MATCHING_NAME = 'name'
MATCHING_CRITERIA = [MATCHING_INDEX, MATCHING_NAME]

# Mapping expressions
# rdkit / openeye have own definitions for these
MATCH_RING = 'match_ring'
MATCH_AROMATICITY = 'match_aromaticity'
MATCH_HYBRIDIZATION = 'match_hybridization'
MATCH_HYBRID_AND_ATOM_TYPE = 'match_hybrid_and_atom_type'
# weak requirements for mapping atoms == more atoms mapped, more in core
# atoms need to match in aromaticity. Same with bonds.
# maps ethane to ethene, CH3 to NH2, but not benzene to cyclohexane
WEAK_ATOM_EXPRESSION = MATCH_AROMATICITY
WEAK_BOND_EXPRESSION = MATCH_AROMATICITY

# default atom expression, requires same aromaticitiy and hybridization
# bonds need to match in bond order
# ethane to ethene wouldn't map, CH3 to NH2 would map but CH3 to HC=O wouldn't
DEFAULT_ATOM_EXPRESSION = MATCH_HYBRIDIZATION
DEFAULT_BOND_EXPRESSION = MATCH_HYBRIDIZATION

# strong requires same hybridization AND the same atom type
# bonds are same as default, require them to match in bond order
STRONG_ATOM_EXPRESSION = MATCH_HYBRID_AND_ATOM_TYPE
STRONG_BOND_EXPRESSION = MATCH_HYBRID_AND_ATOM_TYPE

MappingExpression = namedtuple('MappingExpression', ['atom', 'bond'])
MAP_STRENGTH_DICT = {
    'default': MappingExpression(
        DEFAULT_ATOM_EXPRESSION, DEFAULT_BOND_EXPRESSION),
    'weak': MappingExpression(WEAK_ATOM_EXPRESSION, WEAK_BOND_EXPRESSION),
    'strong': MappingExpression(STRONG_ATOM_EXPRESSION, STRONG_BOND_EXPRESSION)
}

def _score_maps(mol_A, mol_B, maps):
    """ Gives a score for how well each map in a list
    recapitulates the geometry of ligand B.
    If the geometry of ligand B is known, it can identify the closest map,
    if it's not known, it can still be helpful as maps with the same score
    are redundant --- i.e. a flipped phenyl ring.


    Parameters
    ----------
    mol_A : Molecule
        old molecule
    mol_B : Molecule
        new molecule
    maps : list(dict)
        list of maps to score

    Returns
    -------
    list
        list of the distance scores

    """
    coords_A = mol_A.conformers[0]
    coords_B = mol_B.conformers[0]
    # TODO: unwrap these from Quantity
    # Or create utility func that takes two confomers and generates all-to-all distance
    all_to_all = cdist(coords_A, coords_B, 'euclidean')

    mol_B_H = {atom.molecule_atom_index: (atom.atomic_number == 1)
               for atom in mol_B.atoms}
    all_scores = []
    for this_map in maps:
        map_score = 0
        for atom in this_map:
            if not mol_B_H[atom]:  # skip H's - only look at heavy atoms
                map_score += all_to_all[this_map[atom], atom]
        all_scores.append(map_score)
    return all_scores


def _get_mol_atom_map_by_positions(molA, molB, rtol=1e-5, atol=1e-3):
    """
    Return an atom map whereby atoms are mapped if the positions of the atoms are overlapping via np.isclose
    """
    molA_positions = molA.conformers[0]
    molB_positions = molB.conformers[0]
    molB_backward_positions = {val: key for key, val in molB_positions.items()}

    returnable = {}
    for a_idx, a_pos_tup in molA_positions.items():
        match_pos = [idx for idx, b_pos in molB_positions.items()
                     if np.allclose(np.array(a_pos_tup), np.array(b_pos), rtol=rtol, atol=atol)]
        if not len(match_pos) in [0,1]:
            raise Exception(f"there are multiple molB positions with the same coordinates as molA index {a_idx} (by OEMol indexing)")
        if len(match_pos) == 1:
            returnable[match_pos[0]] = a_idx
            #returnable[a_idx] = match_pos[0]

    return returnable


def _scaffold_maps(molA, molB, matching_criterion):
    scaffoldA = get_scaffold(molA)
    scaffoldB = get_scaffold(molB)

    # TODO: atom typing?
    """for atom in scaffoldA.atoms:
        atom.SetIntType(_assign_atom_ring_id(atom))
    for atom in scaffoldB.atoms:
        atom.SetIntType(_assign_atom_ring_id(atom))
    """

    scaffold_maps = _get_all_maps(
        scaffoldA, scaffoldB,
        atom_expr=MATCH_AROMATICITY,
        bond_expr=MATCH_RING,
        external_inttypes=True,
        unique=False,
        matching_criterion=matching_criterion)

    _logger.info('Scaffold has symmetry of %s', len(scaffold_maps))

    if scaffold_maps:
        max_mapped = max([len(m) for m in scaffold_maps])
        _logger.info(f'There are {len(scaffold_maps)} before filtering')
        scaffold_maps = [m for m in scaffold_maps if len(m) == max_mapped]
        _logger.info(f'There are {len(scaffold_maps)} after filtering to remove maps with fewer matches than {max_mapped} atoms')

    return scaffold_maps, scaffoldA, scaffoldB

def _no_common_scaffold_maps(
        molA, molB, *,
        external_inttypes=None, atom_expr=None, bond_expr=None,
        matching_criterion=None):
    _logger.warning('Two molecules are not similar to have a common scaffold')
    _logger.warning('Proceeding with direct mapping of molecules, but please check atom mapping and the geometry of the ligands.')

    # if no commonality with the scaffold, don't use it.
    # why weren't matching arguments carried to these mapping functions? is there an edge case that i am missing?
    # it still doesn't fix the protein sidechain mapping problem
    all_molecule_maps = _get_all_maps(
        molA, molB,
        external_inttypes=external_inttypes,
        atom_expr=atom_expr,
        bond_expr=bond_expr,
        matching_criterion=matching_criterion)
    _logger.info(f'len {all_molecule_maps}')
    for x in all_molecule_maps:
        _logger.info(x)


def _map_with_scaffold(mol, scaffold, matching_criterion):
     scaffold_mol_maps = _get_all_maps(
         mol, scaffold,
         atom_expr=MATCH_HYBRID_AND_ATOM_TYPE,
         bond_expr=DEFAULT_BOND_EXPRESSION,
         matching_criterion=matching_criterion)
     _logger.info(f'{len(scaffold_mol_maps)} scaffold maps')
     scaffold_mol_map = scaffold_mol_maps[0]
     _logger.info(f'Scaffold to mol: {scaffold_mol_map}')
     assert len(scaffold_mol_map) == scaffold.n_atoms, f'Scaffold should be fully contained within the molecule it came from. {len(scaffold_mol_map)} in map, and {scaffold.n_atoms} in scaffold'
     return scaffold_mol_map


def _reset_typing(molA, molB, scaffold_map):
    _logger.warn("Ignoring _reset_typing")
    return
    # reset the IntTypes
    for atom in molA.GetAtoms():
        atom.SetIntType(0)
    for atom in molB.GetAtoms():
        atom.SetIntType(0)

    index = 1
    for scaff_b_id, scaff_a_id in scaffold_map.items():
        for atom in molA.GetAtoms():
            if atom.GetIdx() == scaffold_A_map[scaff_a_id]:
                atom.SetIntType(index)
        for atom in molB.GetAtoms():
            if atom.GetIdx() == scaffold_B_map[scaff_b_id]:
                atom.SetIntType(index)
        index += 1
    for atom in molA.GetAtoms():
        if atom.GetIntType() == 0:
            atom.SetIntType(_assign_atom_ring_id(atom))
    for atom in molB.GetAtoms():
        if atom.GetIntType() == 0:
            atom.SetIntType(_assign_atom_ring_id(atom))


def _maps_with_common_scaffold(
        molA, molB, scaffoldA, scaffoldB, scaffold_maps, *,
        atom_expr=None,
        bond_expr=None,
        allow_ring_breaking=True,
        external_inttypes=None,
        map_strategy=None,
        matching_criterion=None):
    scaffold_A_map = _map_with_scaffold(molA, scaffoldA, matching_criterion)
    scaffold_B_map = _map_with_scaffold(molB, scaffoldB, matching_criterion)
    # now want to find all of the maps
    # for all of the possible scaffold  symmetries
    all_molecule_maps = []
    for scaffold_map in scaffold_maps:
        if external_inttypes is False and allow_ring_breaking is True:
            _reset_typing(molA, molB, scaffold_map)

        molecule_maps = _get_all_maps(
            molA, molB,
            external_inttypes=True,
            atom_expr=atom_expr,
            bond_expr=bond_expr,
            matching_criterion=matching_criterion)
        all_molecule_maps.extend(molecule_maps)
    return all_molecule_maps


def _get_mol_atom_map(molA,
                      molB,
                      atom_expr=None,
                      bond_expr=None,
                      map_strength='default',
                      allow_ring_breaking=True,
                      matching_criterion='index',
                      external_inttypes=False,
                      map_strategy='core'):
    """Find a suitable atom map between two molecules, according
    to the atom_expr, bond_expr or map_strength

    Parameters
    ----------
    molA : Molecule
        old molecule
    molB : Molecule
        new molecule
    atom_expr : int, default=None
        integer corresponding to atom matching, see `perses.openeye.generate_expression`
    bond_expr : int, default=None
        integer corresponding to bond matching, see `perses.openeye.generate_expression`
    map_strength : str, default 'default'
        pre-defined mapping strength that can be one of ['strong', 'default', 'weak']
        this will be ignored if either atom_expr or bond_expr have been defined.
    allow_ring_breaking : bool, optional, default=True
         If False, will check to make sure rings are not being broken or formed.
    matching_criterion : str, default 'index'
         The best atom map is pulled based on some ranking criteria;
         if 'index', the best atom map is chosen based on the map with the maximum number of atomic index matches;
         if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
         else: raise Exception.
         NOTE : the matching criterion pulls patterns and target matches based on indices or names;
                if 'names' is chosen, it is first asserted that the current_mol and the proposed_mol have atoms that are uniquely named
    external_inttypes : bool, default False
        If True, IntTypes already assigned to oemols will be used for mapping, if IntType is in the atom or bond expression.
        Otherwise, IntTypes will be overwritten such as to ensure rings of different sizes are not matched.
    map_strategy : str, default='core'
        determines which map is considered the best and returned
        can be one of ['geometry', 'matching_criterion', 'random', 'weighted-random', 'return-all']
        - core will return the map with the largest number of atoms in the core. If there are multiple maps with the same highest score, then `matching_criterion` is used to tie break
        - geometry uses the coordinates of the molB oemol to calculate the heavy atom distance between the proposed map and the actual geometry
        this can be vital for getting the orientation of ortho- and meta- substituents correct in constrained (i.e. protein-like) environments.
        this is ONLY useful if the positions of ligand B are known and/or correctly aligned.
        - matching_criterion uses the `matching_criterion` flag to pick which of the maps best satisfies a 2D requirement.
        - random will use a random map of those that are possible
        - weighted-random uses a map chosen at random, proportional to how close it is in geometry to ligand B. The same as for 'geometry', this requires the coordinates of ligand B to be meaninful
        - return-all BREAKS THE API as it returns a list of dicts, rather than list. This is intended for development code, not main pipeline.
    Returns
    -------
    dict
        dictionary of scores (keys) and maps (dict)

    """
    assert map_strategy in MAP_STRATEGIES, f'map_strategy cannot be {map_strategy}, it must be one of the allowed options {MAP_STRATEGIES}.'
    _logger.info(f'Using {map_strategy} to chose best atom map')
    assert matching_criterion in MATCHING_CRITERIA, f'matching_criterion cannot be {matching_criterion}, it must be one of the allowed options {MATCHING_CRITERIA}'

    if map_strength is None:
        map_strength = 'default'

    if atom_expr is None:
        _logger.debug(f'No atom expression defined, using map strength : {map_strength}')
        atom_expr = MAP_STRENGTH_DICT[map_strength][0]
    if bond_expr is None:
        _logger.debug(f'No bond expression defined, using map strength : {map_strength}')
        bond_expr = MAP_STRENGTH_DICT[map_strength][1]

    if not external_inttypes or allow_ring_breaking:
        molA = _assign_ring_ids(molA)
        molB = _assign_ring_ids(molB)

    scaffold_maps, scaffoldA, scaffoldB = _scaffold_maps(molA, molB, matching_criterion)

    if len(scaffold_maps) == 0:
        all_molecule_maps = _no_common_scaffold_maps(
            molA, molB,
            external_inttypes=external_inttypes,
            atom_expr=atom_expr,
            bond_expr=bond_expr,
            matching_criterion=matching_criterion)
    else:
        all_molecule_maps = _maps_with_common_scaffold(
            molA, molB, scaffoldA, scaffoldB,
            scaffold_maps,
            atom_expr=atom_expr, bond_expr=bond_expr,
            allow_ring_breaking=allow_ring_breaking,
            external_inttypes=external_inttypes,
            matching_criterion=matching_criterion)
    if not allow_ring_breaking:
        # Filter the matches to remove any that allow ring breaking
        all_molecule_maps = [m for m in all_molecule_maps
                             if preserves_rings(m, molA, molB)]
        _logger.info(f'Checking maps to see if they break rings')

    if len(all_molecule_maps) == 0:
        _logger.warning('No maps found. Try relaxing match criteria or setting allow_ring_breaking to True')
        return None

    if map_strategy == 'return-all':
        _logger.warning('Returning a list of all maps, rather than a dictionary.')
        return all_molecule_maps


    #  TODO - there will be other options that we might want in future here
    #  maybe _get_mol_atom_map  should return a list of maps and then we have
    # a pick_map() function elsewhere?
    # but this would break the API so I'm not doing it now
    if len(all_molecule_maps) == 1:
        _logger.info('Only one map so returning that one')
        return all_molecule_maps[0] #  can this be done in a less ugly way??
    if map_strategy == 'geometry':
        molecule_maps_scores = _remove_redundant_maps(
            molA, molB, all_molecule_maps)
        _logger.info(f'molecule_maps_scores: {molecule_maps_scores.keys()}')
        _logger.info('Returning map with closest geometry satisfaction')
        return molecule_maps_scores[min(molecule_maps_scores)]
    elif map_strategy == 'core':
        core_count = [len(m) for m in all_molecule_maps]
        maximum_core_atoms = max(core_count)
        if core_count.count(maximum_core_atoms) == 1:
            _logger.info('Returning map with most atoms in core')
            return all_molecule_maps[core_count.index(maximum_core_atoms)]
        else:
            best_maps = [m for c, m in zip(core_count, all_molecule_maps) if c == maximum_core_atoms]
            best_map = _score_nongeometric(molA, molB, best_maps, matching_criterion)
            _logger.info(f'{len(best_maps)} have {maximum_core_atoms} core atoms. Using matching_criterion {matching_criterion} to return the best of those')
            return best_map
    elif map_strategy == 'matching_criterion':
        _logger.info('Returning map that best satisfies matching_criterion')
        best_map = _score_nongeometric(molA, molB, all_molecule_maps, matching_criterion)
        return best_map
    elif map_strategy == 'random':
        _logger.info('Returning map at random')
        return np.random.choice(all_molecule_maps.values())
    elif map_strategy == 'weighted-random':
        molecule_maps_scores = _remove_redundant_maps(molA, molB, all_molecule_maps)
        _logger.info(f'molecule_maps_scores: {molecule_maps_scores.keys()}')
        _logger.info('Returning random map proportional to the geometic distance')
        return np.random.choice(molecule_maps_scores.values(),
                                [x**-1 for x in molecule_maps_scores.keys()])

    raise Exception(f'map_strategy {map_strategy} didn\'t match anything')


def _remove_redundant_maps(molA, molB, all_molecule_maps):
    """For a set of maps, it will filter out those that result in
    the same geometries. From redundant maps, one is chosen randomly.

    Parameters
    ----------
    molA : oechem.oemol
        old molecule
    molB : oechem.oemol
        new molecule
    all_molecule_maps : list(dict)
        list of mappings to check for redundancies
        where the maps are molB to molA

    Returns
    -------
    dict
        dictionary of scores (keys) and maps (dict of new-to-old atom indices)
        where the score is the sum of the distances in cartesian space between atoms that have been mapped
        this helps identify when two core atoms that have been assigned to eachother are actually far away.
        maps are molB indices to molA indices.

    """
    scores = _score_maps(molA, molB, all_molecule_maps)
    _logger.info('%s maps are reduced to %s', len(scores), len(set(scores)))
    clusters = {}
    for s, mapping in zip(scores, all_molecule_maps):
        if s not in clusters:
            #  doesn't matter which one is chosen as all is equal
            clusters[s] = mapping
    return clusters


def _get_all_maps(current_mol,
                  proposed_mol,
                  atom_expr=None,
                  bond_expr=None,
                  map_strength='default',
                  allow_ring_breaking=True,
                  external_inttypes=False,
                  unique=True,
                  matching_criterion='index'):
    """Generate all  possible maps between two mols

    Parameters
    ----------
    current_mol : Molecule
        old molecule
    proposed_mol : Molecule
        new molecule
    atom_expr : int, default=None
        integer corresponding to atom matching, see `perses.openeye.generate_expression`
    bond_expr : int, default=None
        integer corresponding to bond matching, see `perses.openeye.generate_expression`
    map_strength : str, default 'default'
        pre-defined mapping strength that can be one of ['strong', 'default', 'weak']
        this will be ignored if either atom_expr or bond_expr have been defined.
    allow_ring_breaking : bool, optional, default=True
         If False, will check to make sure rings are not being broken or formed.
    external_inttypes : bool, default False
        If True, IntTypes already assigned to oemols will be used for mapping, if IntType is in the atom or bond expression.
        Otherwise, IntTypes will be overwritten such as to ensure rings of different sizes are not matched.
    unique : bool, default True
        openeye kwarg which either returns all maps, or filters out redundant ones
    Returns
    -------
    dict
        dictionary of scores (keys) and maps (dict)

    """
    if map_strength is None:
        map_strength = 'default'

    if atom_expr is None:
        _logger.debug(f'No atom expression defined, using map strength : {map_strength}')
        atom_expr = MAP_STRENGTH_DICT[map_strength][0]
    if bond_expr is None:
        _logger.debug(f'No bond expression defined, using map strength : {map_strength}')
        bond_expr = MAP_STRENGTH_DICT[map_strength][1]

    mapping_expr = MappingExpression(atom_expr, bond_expr)
    all_mappings = mcs(current_mol, proposed_mol, mapping_expr)
    return all_mappings


def _score_nongeometric(molA, molB, maps,
                        matching_criterion='index'):
    """
    Given two molecules, returns the mapping of atoms between them using the match with the greatest number of atoms

    Arguments
    ---------
    molA : Molecule
        first molecule
    molB : Molecule
        second molecule
    maps :  list(dict)
        list of maps with which to identify the best match given matching_criterion
    matching_criterion : str, default 'index'
         The best atom map is pulled based on some ranking criteria;
         if 'index', the best atom map is chosen based on the map with the maximum number of atomic index matches;
         if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
         else: raise Exception.
         NOTE : the matching criterion pulls patterns and target matches based on indices or names;
                if 'names' is chosen, it is first asserted that the current_mol and the proposed_mol have atoms that are uniquely named
    Returns
    -------
    matches : list of match
        list of the matches between the molecules, or None if no matches possible

    """

    _logger.info(f'Finding best map using matching_criterion {matching_criterion}')
    max_num_atoms = max([len(m) for m in maps])
    _logger.debug(f"\tthere are {len(maps)} top matches with at most {max_num_atoms} before hydrogen exceptions")

    # now all else is equal; we will choose the map with the highest overlap of atom indices
    index_overlap_numbers = []
    if matching_criterion == 'index':
        for map in maps:
            hit_number = 0
            for key, value in map.items():
                if key == value:
                    hit_number += 1
            index_overlap_numbers.append(hit_number)
    elif matching_criterion == 'name':
        for map in maps:
            hit_number = 0
            map_tuples = list(map.items())
            # is atom hashable?
            atom_map = {  # TODO: looks like it won't get the combinations???
                atom_new: atom_old
                for atom_new, atom_old
                in zip(molB.atoms, molA.atoms)
                if (atom_new.molecule_atom_index,
                    atom_old.molecule_atom_index) in map_tuples}
            for key, value in atom_map.items():
                if key.name == value.name:
                    hit_number += 1
            index_overlap_numbers.append(hit_number)
    else:
        raise Exception(f"the ranking criteria {matching_criterion} is not supported.")

    max_index_overlap_number = max(index_overlap_numbers)
    max_index = index_overlap_numbers.index(max_index_overlap_number)
    map = maps[max_index]

    return map


def _find_closest_map(mol_A, mol_B, maps):
    """ from a list of maps, finds the one that is geometrically the closest match for molecule B

    Parameters
    ----------
    mol_A : oechem.oemol
        The first moleule in the mapping
    mol_B : oechem.oemol
        Second molecule in the mapping
    maps : list(dict)
        A list of maps to search through

    Returns
    -------
    dict
        the single best match from all maps

    """
    if len(maps) == 1:
        return maps[0]
    coords_A = np.zeros(shape=(mol_A.NumAtoms(), 3))
    for i in mol_A.GetCoords():
        coords_A[i] = mol_A.GetCoords()[i]
    coords_B = np.zeros(shape=(mol_B.NumAtoms(), 3))
    for i in mol_B.GetCoords():
        coords_B[i] = mol_B.GetCoords()[i]
    from scipy.spatial.distance import cdist

    all_to_all = cdist(coords_A, coords_B, 'euclidean')

    mol_B_H = {x.GetIdx(): x.IsHydrogen() for x in mol_B.GetAtoms()}
    all_scores = []
    for M in maps:
        map_score = 0
        for atom in M:
            if not mol_B_H[atom]:  # skip H's - only look at heavy atoms
                map_score += all_to_all[M[atom], atom]
        all_scores.append(map_score/len(M))
    _logger.debug(f'Mapping scores: {all_scores}')

    # returning lowest score
    best_map_index = np.argmin(all_scores)
    _logger.debug(f'Returning map index: {best_map_index}')
    return maps[best_map_index]

@staticmethod
def _create_pattern_to_target_map(current_mol, proposed_mol, match, matching_criterion = 'index'):
    """
    Create a dict of {pattern_atom: target_atom}

    Parameters
    ----------
    current_mol : openeye.oechem.oemol object
    proposed_mol : openeye.oechem.oemol object
    match : oechem.OEMCSSearch.Match iter
        entry in oechem.OEMCSSearch.Match object
    matching_criterion : str, default 'index'
        whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
        allowables: ['index', 'name']
    Returns
    -------
    pattern_to_target_map : dict
        {pattern_atom: target_atom}
    """
    if matching_criterion == 'index':
        pattern_atoms = { atom.GetIdx() : atom for atom in current_mol.GetAtoms() }
        target_atoms = { atom.GetIdx() : atom for atom in proposed_mol.GetAtoms() }
        pattern_to_target_map = { pattern_atoms[matchpair.pattern.GetIdx()] : target_atoms[matchpair.target.GetIdx()] for matchpair in match.GetAtoms() }
    elif matching_criterion == 'name':
        pattern_atoms = {atom.GetName(): atom for atom in current_mol.GetAtoms()}
        target_atoms = {atom.GetName(): atom for atom in proposed_mol.GetAtoms()}
        pattern_to_target_map = {pattern_atoms[matchpair.pattern.GetName()]: target_atoms[matchpair.target.GetName()] for matchpair in match.GetAtoms()}
    else:
        raise Exception(f"matching criterion {matching_criterion} is not currently supported")
    return pattern_to_target_map

@staticmethod
def hydrogen_mapping_exceptions(old_mol,
                                new_mol,
                                match,
                                matching_criterion):
    """
    Returns an atom map that omits hydrogen-to-nonhydrogen atom maps AND X-H to Y-H where element(X) != element(Y)
    or aromatic(X) != aromatic(Y)
    Parameters
    ----------
    old_mol : openeye.oechem.oemol object
    new_mol : openeye.oechem.oemol object
    match : openeye.oechem.OEMatchBase
    matching_criterion : str, default

    Returns
    -------
    new_to_old_atom_map : dict
        map of new to old atom indices
    """
    new_to_old_atom_map = {}
    pattern_to_target_map = _create_pattern_to_target_map(
        old_mol, new_mol, match, matching_criterion)
    for pattern_atom, target_atom in pattern_to_target_map.items():
        old_index, new_index = pattern_atom.GetIdx(), target_atom.GetIdx()
        old_atom, new_atom = pattern_atom, target_atom

        #Check if a hydrogen was mapped to a non-hydroden (basically the xor of is_h_a and is_h_b)
        if (old_atom.GetAtomicNum() == 1) != (new_atom.GetAtomicNum() == 1):
            continue

        new_to_old_atom_map[new_index] = old_index

    return new_to_old_atom_map

@staticmethod
def _assign_atom_ring_id(atom, max_ring_size=10):
    """ Returns the int type based on the ring occupancy
    of the atom

    Parameters
    ----------
    atom : oechem.OEAtomBase
        atom to compute integer of
    max_ring_size : int, default = 10
        Largest ring size that will be checked for

    Returns
    -------
    ring_as_base_two : int
        binary integer corresponding to the atoms ring membership
    """
    rings = ''
    for i in range(3, max_ring_size+1): #  smallest feasible ring size is 3
        rings += str(int(oechem.OEAtomIsInRingSize(atom, i)))
    ring_as_base_two = int(rings, 2)
    return ring_as_base_two

@staticmethod
def _assign_bond_ring_id(bond, max_ring_size=10):
    """ Returns the int type based on the ring occupancy
    of the bond

    Parameters
    ----------
    bond : oechem.OEBondBase
        atom to compute integer of
    max_ring_size : int, default = 10
        Largest ring size that will be checked for

    Returns
    -------
    ring_as_base_two : int
        binary integer corresponding to the bonds ring membership
    """
    rings = ''
    for i in range(3, max_ring_size+1): #  smallest feasible ring size is 3
        rings += str(int(oechem.OEBondIsInRingSize(bond, i)))
    ring_as_base_two = int(rings, 2)
    return ring_as_base_two


def _assign_ring_ids(molecule, max_ring_size=10):
    """ Sets the Int of each atom in the mol to a number
    corresponding to the ring membership of that atom

    Parameters
    ----------
    molecule : Molecule
        mol to assign ring ID to
    max_ring_size : int, default = 10
        Largest ring size that will be checked for

    Returns
    -------
    """
    return molecule
    # TODO: how to assign atom and bond types in openforcefield?
    for atom in molecule.atoms:
        atom.SetIntType(
            _assign_atom_ring_id(atom, max_ring_size=max_ring_size))
    for bond in molecule.bonds:
        bond.SetIntType(
            _assign_bond_ring_id(bond, max_ring_size=max_ring_size))
    return molecule


def _assign_distance_ids(old_mol, new_mol, distance=0.3):
    """ Gives atoms in both molecules matching Int numbers if they are close
    to each other. This should ONLY be  used if the geometry (i.e. binding mode)
    of both molecules are known, and they are aligned to the same frame of reference.

    this function is invariant to which is passed in as {old|new}_mol
    Parameters
    ----------
    old_mol : oechem.OEMol
        first molecule to compare
    new_mol : oechem.OEMol
        second molecule to compare
    distance : float, default = 0.3
        Distance (in angstrom) that two atoms need to be closer than to be
        labelled as the same.

    Returns
    -------
    copies of old_mol and new_mol, with IntType set according to inter-molecular distances
    """
    _logger.info(f'Using a distance of {distance} to force the mapping of close atoms')
    from scipy.spatial.distance import cdist
    unique_integer = 1
    for atomA, coordsA in zip(old_mol.GetAtoms(), old_mol.GetCoords().values()):
        for atomB, coordsB in zip(new_mol.GetAtoms(), new_mol.GetCoords().values()):
            distances_ij = cdist([coordsA], [coordsB], 'euclidean')[0]
            if distances_ij < distance:
                atomA.SetIntType(unique_integer)
                atomB.SetIntType(unique_integer)
                unique_integer += 1
    return old_mol, new_mol


def preserves_rings(new_to_old_map, current, proposed):
    """Returns True if the transformation allows ring
    systems to be broken or created."""
    if _breaks_rings_in_transformation(new_to_old_map,
                                       proposed):
        return False
    old_to_new_map = {i: j for j,i in new_to_old_map.items()}
    if _breaks_rings_in_transformation(old_to_new_map,
                                       current):
        return False
    return True


def preserve_chirality(current_mol, proposed_mol, new_to_old_atom_map):
    """
    filters the new_to_old_atom_map for chirality preservation
    The current scheme is implemented as follows:
    for atom_new, atom_old in new_to_old.items():
        if atom_new is R/S and atom_old is undefined:
            # we presume that one of the atom neighbors is being changed, so map it accordingly
        elif atom_new is undefined and atom_old is R/S:
            # we presume that one of the atom neighbors is not being mapped, so map it accordingly
        elif atom_new is R/S and atom_old is R/S:
            # we presume nothing is changing
        elif atom_new is S/R and atom_old is R/S:
            # we presume that one of the neighbors is changing
            # check if all of the neighbors are being mapped:
                if True, flip two
                else: do nothing
    """
    pattern_atoms = { atom.GetIdx() : atom for atom in current_mol.GetAtoms() }
    target_atoms = { atom.GetIdx() : atom for atom in proposed_mol.GetAtoms() }
    # _logger.warning(f"\t\t\told oemols: {pattern_atoms}")
    # _logger.warning(f"\t\t\tnew oemols: {target_atoms}")
    copied_new_to_old_atom_map = copy.deepcopy(new_to_old_atom_map)
    _logger.info(new_to_old_atom_map)

    for new_index, old_index in new_to_old_atom_map.items():

        if target_atoms[new_index].IsChiral() and not pattern_atoms[old_index].IsChiral():
            #make sure that not all the neighbors are being mapped
            #get neighbor indices:
            neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
            if all(nbr in set(list(new_to_old_atom_map.keys())) for nbr in neighbor_indices):
                _logger.warning(f"the atom map cannot be reconciled with chirality preservation!  It is advisable to conduct a manual atom map.")
                return {}
            else:
                #try to remove a hydrogen
                hydrogen_maps = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms() if atom.GetAtomicNum() == 1]
                mapped_hydrogens = [_idx for _idx in hydrogen_maps if _idx in list(new_to_old_atom_map.keys())]
                if mapped_hydrogens != []:
                    del copied_new_to_old_atom_map[mapped_hydrogens[0]]
                else:
                    _logger.warning(f"there may be a geometry problem!  It is advisable to conduct a manual atom map.")
        elif not target_atoms[new_index].IsChiral() and pattern_atoms[old_index].IsChiral():
            #we have to assert that one of the neighbors is being deleted
            neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
            if any(nbr_idx not in list(new_to_old_atom_map.keys()) for nbr_idx in neighbor_indices):
                pass
            else:
                _logger.warning(f"the atom map cannot be reconciled with chirality preservation since no hydrogens can be deleted!  It is advisable to conduct a manual atom map.")
                return {}
        elif target_atoms[new_index].IsChiral() and pattern_atoms[old_index].IsChiral() and oechem.OEPerceiveCIPStereo(current_mol, pattern_atoms[old_index]) == oechem.OEPerceiveCIPStereo(proposed_mol, target_atoms[new_index]):
            #check if all the atoms are mapped
            neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
            if all(nbr in set(list(new_to_old_atom_map.keys())) for nbr in neighbor_indices):
                pass
            else:
                _logger.warning(f"the atom map cannot be reconciled with chirality preservation since all atom neighbors are being mapped!  It is advisable to conduct a manual atom map.")
                return {}
        elif target_atoms[new_index].IsChiral() and pattern_atoms[old_index].IsChiral() and oechem.OEPerceiveCIPStereo(current_mol, pattern_atoms[old_index]) != oechem.OEPerceiveCIPStereo(proposed_mol, target_atoms[new_index]):
            neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
            if all(nbr in set(list(new_to_old_atom_map.keys())) for nbr in neighbor_indices):
                _logger.warning(f"the atom map cannot be reconciled with chirality preservation since all atom neighbors are being mapped!  It is advisable to conduct a manual atom map.")
                return {}
            else:
                #try to remove a hydrogen
                hydrogen_maps = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms() if atom.GetAtomicNum() == 1]
                mapped_hydrogens = [_idx for _idx in hydrogen_maps if _idx in list(new_to_old_atom_map.keys())]
                if mapped_hydrogens != []:
                    del copied_new_to_old_atom_map[mapped_hydrogens[0]]
                else:
                    _logger.warning(f"there may be a geometry problem.  It is advisable to conduct a manual atom map.")

    return copied_new_to_old_atom_map #was this really an indentation error?


def _breaks_rings_in_transformation(atom_map, current):
        """Return True if the transformation from molecule1 to molecule2 breaks rings.

        Parameters
        ----------
        atom_map : dict of Atom : Atom
            atom_map[molecule1_atom] is the corresponding molecule2 atom
        current : oechem.OEMol
            Initial molecule whose rings are to be checked for not being broken

        Returns
        -------
        bool
        """
        for cycle in _enumerate_cycle_basis(current):
            cycle_size = len(cycle)
            # first check that ALL of the ring is in the map or out
            atoms_in_cycle = set(
                [bond.atom1_index for bond in cycle] +
                [bond.atom2_index for bond in cycle])
            number_of_cycle_atoms_mapped = 0
            for atom in atoms_in_cycle:
                if atom in atom_map:
                    number_of_cycle_atoms_mapped += 1
            _logger.info(number_of_cycle_atoms_mapped)
            if number_of_cycle_atoms_mapped == 0:
                # none of the ring is mapped - ALL unique, so continue
                continue
            if number_of_cycle_atoms_mapped != len(atoms_in_cycle):
                return True # not all atoms in ring are mapped
        return False  # no rings in molecule1 are broken in molecule2


def _enumerate_cycle_basis(molecule):
    """Enumerate a closed cycle basis of bonds in molecule.

    This uses cycle_basis from NetworkX:
    https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis

    Parameters
    ----------
    molecule : Molecule
        The molecule for a closed cycle basis of Bonds is to be identified

    Returns
    -------
    bond_cycle_basis : list of list of OEBond
        bond_cycle_basis[cycle_index] is a list of OEBond objects that define a cycle in the basis
        You can think of these as the minimal spanning set of ring systems to check.
    """
    g = nx.Graph()
    for atom in molecule.atoms:
        g.add_node(atom.molecule_atom_index)
    for bond in molecule.bonds:
        g.add_edge(bond.atom1_index, bond.atom2_index, bond=bond)
    bond_cycle_basis = list()
    for cycle in nx.cycle_basis(g):
        bond_cycle = list()
        for i in range(len(cycle)):
            atom_index_1 = cycle[i]
            atom_index_2 = cycle[(i+1) % len(cycle)]
            edge = g.edges[atom_index_1, atom_index_2]
            bond = edge['bond']
            bond_cycle.append(bond)
        bond_cycle_basis.append(bond_cycle)
    return bond_cycle_basis


def rank_degenerate_maps(matches, current, proposed):
    raise Exception("Not implemented")


class AtomMapper:
    """ Generates map between two molecules.
    As this doesn't generate a system, it will not be
    accurate for whether hydrogens are mapped or not.
    It also doesn't check that this is feasible to simulate,
    but is just helpful for testing options.

    Parameters
    ----------
    current_molecule : Mol
        starting molecule
    proposed_molecule : Mol
        molecule to move to
    atom_expr : oechem.OEExprOpts
        atom requirement to consider two bonds as the same
    bond_expr : oechem.OEExprOpts
        bond requirement to consider two bonds as the same
    allow_ring_breaking : bool, default=True
        Wether or not to allow ring breaking in map
    map_strategy : str, default = 'core'

    Attributes
    ----------
    map : type
        Description of attribute `map`.
    _get_mol_atom_map : type
        Description of attribute `_get_mol_atom_map`.
    current_molecule
    proposed_molecule
    atom_expr
    bond_expr
    allow_ring_breaking
    map_strategy

    """
    def save_atom_mapping(self, filename='atom_map.png'):
        render_atom_mapping(
            filename, self.current_mol, self.proposed_mol, self.atom_map)
