"""

Utility functions for simulations using openforcefield toolkit

"""

__author__ = 'John D. Chodera'


#from openeye import oechem, oegraphsim
#from openmoltools.openeye import generate_conformers
from simtk import openmm, unit
from simtk.openmm import app
import simtk.unit as unit
import numpy as np
import logging

from openforcefield.topology import Molecule
from openforcefield.utils import get_data_file_path
from openmmforcefields.generators import SystemGenerator
from perses.utils.smallmolecules import sanitizeSMILES, describe_mol,\
    smiles_to_mol, createMolFromSDF, subset_molecule

logging.basicConfig(level=logging.NOTSET)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def system_generator_wrapper(
        mols,
        barostat = None,
        forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
        forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus},
        nonperiodic_forcefield_kwargs = {'nonbondedMethod': app.NoCutoff},
        small_molecule_forcefield = 'gaff-2.11',
        **kwargs):
    """
    make a system generator (vacuum) for a small molecule

    Parameters
    ----------
    mols : list of openforcefield.topology.Molecule
        mols
    barostat : openmm.MonteCarloBarostat, default None
        barostat
    forcefield_files : list of str
        pointers to protein forcefields and solvent
    forcefield_kwargs : dict
        dict of forcefield_kwargs
    nonperiodic_forcefield_kwargs : dict
        dict of args for non-periodic system
    small_molecule_forcefield : str
        pointer to small molecule forcefield to use

    Returns
    -------
    system_generator : openmmforcefields.generators.SystemGenerator
    """
    system_generator = SystemGenerator(
        forcefields=forcefield_files,
        barostat=barostat,
        forcefield_kwargs=forcefield_kwargs,
        nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
        small_molecule_forcefield=small_molecule_forcefield,
        molecules=mols, cache=None)
    return system_generator


def extractPositionsFromMol(molecule, units=unit.angstrom):
    """
    Get a molecules coordinates from an openff mol

    Parameters
    ----------
    molecule : openforcefield.topology.Molecule object
    units : simtk.unit, default angstrom

    Returns
    -------
    positions : np.array
    """
    raise Exception("Should convert units")
    return molecule.conformers[0]


def giveOpenmmPositionsToMol(positions, molecule):
    """
    Replace Mol positions with openmm format positions

    Parameters
    ----------
    positions : openmm.topology.positions
    molecule : openforcefield.topology.Molecule object

    Returns
    -------
    molecule : openforcefield.topology.Molecule
        molecule with updated positions

    """
    assert molecule.n_atoms == len(positions), "Number of openmm positions does not match number of atoms in Mol object"
    molecule.conformers[0] = positions

    return molecule  # TODO: Why return the input molecule?


def mol_to_omm_ff(molecule, system_generator):
    """
    Convert an openforcefield.topology.Molecule to a openmm system, positions and topology

    Parameters
    ----------
    mol : openforcefield.topology.Molecule object
        input molecule to convert
    system_generator : openmmforcefields.generators.SystemGenerator

    Returns
    -------
    system : openmm.system
    positions : openmm.positions
    topology : openmm.topology

    """

    topology = molecule.to_topology().to_openmm()
    system = system_generator.create_system(topology)
    positions = extractPositionsFromMol(molecule)
    # TODO: positions are trivially extracted from topology. why two arguments?
    return system, positions, topology


def createSystemFromSMILES(smiles, title='MOL', **system_generator_kwargs):
    """
    Create an openmm system from a smiles string

    Parameters
    ----------
    smiles : str
        smiles string of molecule

    Returns
    -------
    molecule : openforcefield.topology.Molecule
        Mol molecule
    system : openmm.System object
        OpenMM system
    positions : [n,3] np.array of floats
        Positions
    topology : openmm.app.Topology object
        Topology
    """
    # clean up smiles string

    smiles = sanitizeSMILES([smiles])
    smiles = smiles[0]

    # Create Mol
    molecule = smiles_to_mol(smiles, title=title)
    system_generator = system_generator_wrapper([molecule], **system_generator_kwargs)

    # generate openmm system, positions and topology
    system, positions, topology = mol_to_omm_ff(molecule, system_generator)

    return (molecule, system, positions, topology)


# def calculate_mol_similarity(molA, molB): DELETED, unused


def generate_expression(list):
    """Turns a list of strings into an atom or bond expression
    This allows us to pass in matching expressions in the input .yaml
    Note: strings are case sensitive

    >>> atom_expr = generate_expression("Hybridization", "IntType")

    Parameters
    ----------
    list : list of strings
        List of strings

    Returns
    -------
    integer
        Integer that openeye magically understands for matching expressions

    """
    total_expr = 0

    for string in list:
        try:
            expr = getattr(oechem, f'OEExprOpts_{string}')
        except AttributeError:
            raise Exception(f'{string} not recognised, no expression of oechem.OEExprOpts_{string}.\
            This is case sensitive, so please check carefully and see , \
            https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemConstants/OEExprOpts.html\
            for options')
        # doing bitwise OR check
        total_expr = total_expr | expr

    return total_expr


def get_scaffold(molecule, adjustHcount=False):
    """
    Takes a mol and returns
    a mol of the scaffold

    The scaffold is a molecule where all the atoms that are not in rings, and are not linkers between rings.
    double bonded atoms exo to a ring are included as ring atoms

    This function has been completely taken from openeye's extractscaffold.py script
    https://docs.eyesopen.com/toolkits/python/oechemtk/oechem_examples/oechem_example_extractscaffold.html#section-example-oechem-extractscaffold
    Parameters
    ----------
    molecule : Molecule
        entire molecule to get the scaffold of
    adjustHcount : bool, default=False
        add/remove hydrogens to satisfy valence of scaffold


    Returns
    -------
    Molecule
        scaffold mol of the input mol. New mol.
    """
    def TraverseForRing(visited, atom):
        visited.add(atom.molecule_atom_index)

        for nbor in atom.bonded_atoms:
            if nbor.molecule_atom_index not in visited:
                if nbor.is_in_ring:
                    return True

                if TraverseForRing(visited, nbor):
                    return True

        return False

    def DepthFirstSearchForRing(root, nbor):
        visited = set([root.molecule_atom_index])
        return TraverseForRing(visited, nbor)

    def is_in_scaffold(atom):
        if atom.is_in_ring:
            return True

        count = 0
        for nbor in atom.bonded_atoms:
            if DepthFirstSearchForRing(atom, nbor):
                count += 1

        return count > 1

    included_atom_indexes = {
        atom.molecule_atom_index
        for atom in molecule.atoms if is_in_scaffold(atom)}

    dst = subset_molecule(molecule, included_atom_indexes)
    # TODO: how to handle adjusthcount True?
    if adjustHcount:
        dst = Molecule.from_smiles(
            dst.to_smiles(explicit_hydrogens=False))
    return dst
