"""
Unit tests for NCMC switching engine.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################
import copy
from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from functools import partial
from pkg_resources import resource_filename
from perses.rjmc import geometry
from perses.rjmc.topology_proposal import SystemGenerator, TopologyProposal, SmallMoleculeSetProposalEngine
from openeye import oechem
if sys.version_info >= (3, 0):
    from io import StringIO
    from subprocess import getstatusoutput
else:
    from cStringIO import StringIO
    from commands import getstatusoutput
from openmmtools.constants import kB
from openmmtools import alchemy, states

################################################################################
# CONSTANTS
################################################################################

temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
ENERGY_THRESHOLD = 1e-1

################################################################################
# UTILITIES
################################################################################]

# TODO: Move some of these utility routines to openmoltools.

class NaNException(Exception):
    pass

def quantity_is_finite(quantity):
    """
    Check that elements in quantity are all finite.

    Parameters
    ----------
    quantity : simtk.unit.Quantity
        The quantity to check

    Returns
    -------
    is_finite : bool
        If quantity is finite, returns True; otherwise False.

    """
    if np.any( np.isnan( np.array(quantity / quantity.unit) ) ):
        return False
    return True

def compare_at_lambdas(context, functions):
    """
    Compare the energy components at all lambdas = 1 and 0.
    """

    #first, set all lambdas to 0
    for parm in functions.keys():
        context.setParameter(parm, 0.0)

    energy_components_0 = compute_potential_components(context)

    for parm in functions.keys():
        context.setParameter(parm, 1.0)

    energy_components_1 = compute_potential_components(context)

    print("-----------------------")
    print("Energy components at lambda=0")

    for i in range(len(energy_components_0)):
        name, value = energy_components_0[i]
        print("%s\t%s" % (name, str(value)))

    print("-----------------------")
    print("Energy components at lambda=1")

    for i in range(len(energy_components_1)):
        name, value = energy_components_1[i]
        print("%s\t%s" % (name, str(value)))

    print("------------------------")



def get_atoms_with_undefined_stereocenters(molecule, verbose=False):
    """
    Return list of atoms with undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to check.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Returns
    -------
    atoms : list of openeye.oechem.OEAtom
        List of atoms with undefined stereochemistry.

    """
    from openeye.oechem import OEAtomStereo_Undefined, OEAtomStereo_Tetrahedral
    undefined_stereocenters = list()
    for atom in molecule.GetAtoms():
        chiral = atom.IsChiral()
        stereo = OEAtomStereo_Undefined
        if atom.HasStereoSpecified(OEAtomStereo_Tetrahedral):
            v = list()
            for nbr in atom.GetAtoms():
                v.append(nbr)
            stereo = atom.GetStereo(v, OEAtomStereo_Tetrahedral)

        if chiral and (stereo == OEAtomStereo_Undefined):
            undefined_stereocenters.append(atom)
            if verbose:
                print("Atom %d (%s) of molecule '%s' has undefined stereochemistry (chiral=%s, stereo=%s)." % (atom.GetIdx(), atom.GetName(), molecule.GetTitle(), str(chiral), str(stereo)))

    return undefined_stereocenters

def has_undefined_stereocenters(molecule, verbose=False):
    """
    Check if given molecule has undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to check.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    TODO
    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Returns
    -------
    result : bool
        True if molecule has undefined stereocenters.

    Examples
    --------
    Enumerate undefined stereocenters
    >>> smiles = "[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]"
    >>> from openeye.oechem import OEGraphMol, OESmilesToMol
    >>> molecule = OEGraphMol()
    >>> OESmilesToMol(molecule, smiles)
    >>> print has_undefined_stereocenters(smiles)
    True

    """
    #TODO move to utils
    atoms = get_atoms_with_undefined_stereocenters(molecule, verbose=verbose)
    if len(atoms) > 0:
        return True

    return False

def enumerate_undefined_stereocenters(molecule, verbose=False):
    """
    Check if given molecule has undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule whose stereocenters are to be expanded.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    Returns
    -------
    molecules : list of OEMol
        The molecules with fully defined stereocenters.

    TODO
    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Examples
    --------
    Enumerate undefined stereocenters
    >>> smiles = "[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]"
    >>> from openeye.oechem import OEGraphMol, OESmilesToMol
    >>> molecule = OEGraphMol()
    >>> OESmilesToMol(molecule, smiles)
    >>> molecules = enumerate_undefined_stereocenters(smiles)
    >>> len(molecules)
    2

    """
    #TODO move to utils
    from openeye.oechem import OEAtomStereo_RightHanded, OEAtomStereo_LeftHanded, OEAtomStereo_Tetrahedral
    from itertools import product

    molecules = list()
    atoms = get_atoms_with_undefined_stereocenters(molecule, verbose=verbose)
    for stereocenters in product([OEAtomStereo_RightHanded, OEAtomStereo_LeftHanded], repeat=len(atoms)):
        for (index,atom) in enumerate(atoms):
            neighbors = list()
            for neighbor in atom.GetAtoms():
                neighbors.append(neighbor)
            atom.SetStereo(neighbors, OEAtomStereo_Tetrahedral, stereocenters[index])
        molecules.append(molecule.CreateCopy())

    return molecules

def test_sanitizeSMILES():
    """
    Test SMILES sanitization.
    """
    from perses.utils.smallmolecules import sanitizeSMILES
    smiles_list = ['CC', 'CCC', '[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]']

    sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='drop')
    if len(sanitized_smiles_list) != 2:
        raise Exception("Molecules with undefined stereochemistry are not being properly dropped (size=%d)." % len(sanitized_smiles_list))

    sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='expand')
    if len(sanitized_smiles_list) != 4:
        raise Exception("Molecules with undefined stereochemistry are not being properly expanded (size=%d)." % len(sanitized_smiles_list))

    # Check that all molecules can be round-tripped.
    from openeye.oechem import OEGraphMol, OESmilesToMol, OECreateIsoSmiString
    for smiles in sanitized_smiles_list:
        molecule = OEGraphMol()
        OESmilesToMol(molecule, smiles)
        isosmiles = OECreateIsoSmiString(molecule)
        if (smiles != isosmiles):
            raise Exception("Molecule '%s' was not properly round-tripped (result was '%s')" % (smiles, isosmiles))

def compute_potential(system, positions, platform=None):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    system : simtk.openmm.System
        The system object to check.
    positions : simtk.unit.Quantity of size (natoms,3) with units compatible with nanometers
        The positions to check.
    platform : simtk.openmm.Platform, optional, default=none
        If specified, this platform will be used.

    """
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    if platform is not None:
        context = openmm.Context(system, integrator, platform)
    else:
        context = openmm.Context(system, integrator)
    context.setPositions(positions)
    context.applyConstraints(integrator.getConstraintTolerance())
    potential = context.getState(getEnergy=True).getPotentialEnergy()
    del context, integrator
    if np.isnan(potential / unit.kilocalories_per_mole):
        raise NaNException("Potential energy is NaN")
    return potential

def compute_potential_components(context):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    context : simtk.openmm.Context
        The context from which to extract, System, parameters, and positions.

    """
    # Make a deep copy of the system.
    import copy
    system = context.getSystem()
    system = copy.deepcopy(system)
    # Get positions.
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Get Parameters
    parameters = context.getParameters()
    # Segregate forces.
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
    # Create new Context.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for (parameter, value) in parameters.items():
        context.setParameter(parameter, value)
    energy_components = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        forcename = force.__class__.__name__
        groups = 1<<index
        potential = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        energy_components.append((forcename, potential))
    del context, integrator
    return energy_components

def check_system(system):
    """
    Check OpenMM System object for pathologies, like duplicate atoms in torsions.

    Parameters
    ----------
    system : simtk.openmm.System

    """
    forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
    force = forces['PeriodicTorsionForce']
    for index in range(force.getNumTorsions()):
        [i, j, k, l, periodicity, phase, barrier] = force.getTorsionParameters(index)
        if len(set([i,j,k,l])) < 4:
            msg  = 'Torsion index %d of self._topology_proposal.new_system has duplicate atoms: %d %d %d %d\n' % (index,i,j,k,l)
            msg += 'Serialzed system to system.xml for inspection.\n'
            raise Exception(msg)
    from simtk.openmm import XmlSerializer
    serialized_system = XmlSerializer.serialize(system)
    outfile = open('system.xml', 'w')
    outfile.write(serialized_system)
    outfile.close()

def generate_endpoint_thermodynamic_states(system: openmm.System, topology_proposal: TopologyProposal):
    """
    Generate endpoint thermodynamic states for the system

    Parameters
    ----------
    system : openmm.System
        System object corresponding to thermodynamic state
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        TopologyProposal representing transformation

    Returns
    -------
    nonalchemical_zero_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda zero endpoint
    nonalchemical_one_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda one endpoint
    lambda_zero_thermodynamic_state : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda zero
    lambda_one_thermodynamic_State : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda one
    """
    #create the thermodynamic state
    from perses.annihilation.lambda_protocol import RelativeAlchemicalState

    lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(system)
    lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

    #ensure their states are set appropriately
    lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
    lambda_one_alchemical_state.set_alchemical_parameters(1.0)

    check_system(system)

    #create the base thermodynamic state with the hybrid system
    thermodynamic_state = states.ThermodynamicState(system, temperature=temperature)

    #Create thermodynamic states for the nonalchemical endpoints
    nonalchemical_zero_thermodynamic_state = states.ThermodynamicState(topology_proposal.old_system, temperature=temperature)
    nonalchemical_one_thermodynamic_state = states.ThermodynamicState(topology_proposal.new_system, temperature=temperature)

    #Now create the compound states with different alchemical states
    lambda_zero_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_zero_alchemical_state])
    lambda_one_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_one_alchemical_state])

    return nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state


def generate_vacuum_topology_proposal(current_mol_name="benzene", proposed_mol_name="toluene", forcefield_kwargs=None, system_generator_kwargs=None, propose_geometry = True):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet from two IUPAC molecule names.

    Constraints are added to the system by default. To override this, set ``forcefield_kwargs = None``.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule
    forcefield_kwargs : dict, optional, default=None
        Additional arguments to ForceField in addition to
        'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff
    system_generator_kwargs : dict, optional, default=None
        Dict passed onto SystemGenerator

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """

    from perses.utils.openeye import extractPositionsFromOEMol
    from openmoltools.openeye import iupac_to_oemol, generate_conformers
    from perses.utils.data import get_data_filename
    from perses.utils.smallmolecules import render_atom_mapping

    gaff_filename = get_data_filename('data/gaff.xml')
    default_forcefield_kwargs = {'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff, 'constraints' : app.HBonds}
    forcefield_kwargs = default_forcefield_kwargs.update(forcefield_kwargs) if (forcefield_kwargs is not None) else default_forcefield_kwargs
    system_generator_kwargs = system_generator_kwargs if (system_generator_kwargs is not None) else dict()
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'],
        forcefield_kwargs=forcefield_kwargs,
        **system_generator_kwargs)

    old_oemol = iupac_to_oemol(current_mol_name)
    old_oemol = generate_conformers(old_oemol,max_confs=1)
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    old_topology = generateTopologyFromOEMol(old_oemol)

    #extract old positions and turn to nanometers
    old_positions = extractPositionsFromOEMol(old_oemol)
    old_positions = old_positions.in_units_of(unit.nanometers)

    old_smiles = oechem.OEMolToSmiles(old_oemol)
    old_system = system_generator.build_system(old_topology)

    new_oemol = iupac_to_oemol(proposed_mol_name)
    new_oemol = generate_conformers(new_oemol,max_confs=1)

    new_smiles = oechem.OEMolToSmiles(new_oemol)

    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [old_smiles, new_smiles], system_generator, residue_name=current_mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(old_system, old_topology, current_mol=old_oemol, proposed_mol=new_oemol)

    # show atom mapping
    filename = str(current_mol_name)+str(proposed_mol_name)+'.pdf'
    render_atom_mapping(filename, old_oemol, new_oemol, topology_proposal.new_to_old_atom_map)

    if propose_geometry:
        #generate new positions with geometry engine
        new_positions, _ = geometry_engine.propose(topology_proposal, old_positions, beta)
    else:
        new_positions, _ = None, None

    # DEBUG: Zero out bonds and angles for one system
    #print('Zeroing bonds of old system')
    #for force in topology_proposal.old_system.getForces():
    #    if force.__class__.__name__ == 'HarmonicAngleForce':
    #        for index in range(force.getNumAngles()):
    #            p1, p2, p3, angle, K = force.getAngleParameters(index)
    #            K *= 0.0
    #            force.setAngleParameters(index, p1, p2, p3, angle, K)
    #    if False and force.__class__.__name__ == 'HarmonicBondForce':
    #        for index in range(force.getNumBonds()):
    #            p1, p2, r0, K = force.getBondParameters(index)
    #            K *= 0.0
    #            force.setBondParameters(index, p1, p2, r0, K)

    # DEBUG : Box vectors
    #box_vectors = np.eye(3) * 100 * unit.nanometers
    #topology_proposal.old_system.setDefaultPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
    #topology_proposal.new_system.setDefaultPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])

    return topology_proposal, old_positions, new_positions

def  generate_solvated_hybrid_test_topology(current_mol_name="naphthalene", proposed_mol_name="benzene", vacuum = False):
    """
    Arguments
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule
    vacuum: bool (default False)
        whether to render a vacuum or solvated topology_proposal

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from openeye import oechem
    from openmoltools.openeye import iupac_to_oemol, generate_conformers
    from openmoltools import forcefield_generators
    import perses.utils.openeye as openeye
    from perses.utils.data import get_data_filename
    from perses.rjmc.topology_proposal import TopologyProposal, SystemGenerator, SmallMoleculeSetProposalEngine
    import simtk.unit as unit

    old_oemol, new_oemol = iupac_to_oemol(current_mol_name), iupac_to_oemol(proposed_mol_name)

    old_smiles = oechem.OECreateSmiString(old_oemol,oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
    new_smiles = oechem.OECreateSmiString(new_oemol,oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)

    old_oemol, old_system, old_positions, old_topology = openeye.createSystemFromSMILES(old_smiles, title = "MOL")

    #correct the old positions
    old_positions = openeye.extractPositionsFromOEMol(old_oemol)
    old_positions = old_positions.in_units_of(unit.nanometers)


    new_oemol, new_system, new_positions, new_topology = openeye.createSystemFromSMILES(new_smiles, title = "NEW")


    ffxml = forcefield_generators.generateForceFieldFromMolecules([old_oemol, new_oemol])

    old_oemol.SetTitle('MOL'); new_oemol.SetTitle('MOL')

    old_topology = forcefield_generators.generateTopologyFromOEMol(old_oemol)
    new_topology = forcefield_generators.generateTopologyFromOEMol(new_oemol)

    if not vacuum:
        nonbonded_method = app.PME
        barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, 300.0*unit.kelvin, 50)
    else:
        nonbonded_method = app.NoCutoff
        barostat = None

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    system_generator = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'tip3p.xml'],barostat = barostat, forcefield_kwargs={'removeCMMotion': False,'nonbondedMethod': nonbonded_method,'constraints' : app.HBonds, 'hydrogenMass' : 4.0*unit.amu})
    system_generator._forcefield.loadFile(StringIO(ffxml))

    proposal_engine = SmallMoleculeSetProposalEngine([old_smiles, new_smiles], system_generator, residue_name = 'MOL')

    if not vacuum:
        #now to solvate
        modeller = app.Modeller(old_topology, old_positions)
        hs = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H'] and atom.residue.name not in ['MOL','OLD','NEW']]
        modeller.delete(hs)
        modeller.addHydrogens(forcefield=system_generator._forcefield)
        modeller.addSolvent(system_generator._forcefield, model='tip3p', padding=9.0*unit.angstroms)
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()
        solvated_positions = unit.quantity.Quantity(value = np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit = unit.nanometers)
        solvated_system = system_generator.build_system(solvated_topology)
        solvated_system.addForce(barostat)

        #now to create proposal
        top_proposal = proposal_engine.propose(solvated_system, solvated_topology, old_oemol)

        return top_proposal, solvated_positions, None

    else:
        vacuum_system = system_generator.build_system(old_topology)
        top_proposal = proposal_engine.propose(vacuum_system, old_topology, old_oemol)
        return top_proposal, old_positions, None

def generate_vacuum_hostguest_proposal(current_mol_name="B2", proposed_mol_name="MOL"):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    from openmoltools import forcefield_generators
    from openmmtools import testsystems

    from openmoltools.openeye import smiles_to_oemol, generate_conformers
    from perses.utils.data import get_data_filename

    host_guest = testsystems.HostGuestVacuum()
    unsolv_old_system, old_positions, top_old = host_guest.system, host_guest.positions, host_guest.topology

    ligand_topology = [res for res in top_old.residues()]
    current_mol = forcefield_generators.generateOEMolFromTopologyResidue(ligand_topology[1]) # guest is second residue in topology
    proposed_mol = smiles_to_oemol('C1CC2(CCC1(CC2)C)C')
    proposed_mol = generate_conformers(proposed_mol,max_confs=1)

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff})
    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=current_mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old, current_mol=current_mol, proposed_mol=proposed_mol)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, old_positions, beta)

    return topology_proposal, old_positions, new_positions

def validate_rjmc_work_variance(top_prop, positions, geometry_method = 0, num_iterations = 10, md_steps = 250, compute_timeseries = False, prespecified_conformers = None):
    """
    Arguments
    ----------
    top_prop : perses.rjmc.topology_proposal.TopologyProposal object
        topology_proposal
    geometry_method : int
        which geometry proposal method to use
            0: neglect_angles = True (this is supposed to be the zero-variance method)
            1: neglect_angles = False (this will accumulate variance)
            2: use_sterics = True (this is experimental)
    num_iterations: int
        number of times to run md_steps integrator
    md_steps: int
        number of md_steps to run in each num_iteration
    compute_timeseries = bool (default False)
        whether to use pymbar detectEquilibration and subsampleCorrelated data from the MD run (the potential energy is the data)
    prespecified_conformers = None or unit.Quantity(np.array([num_iterations, system.getNumParticles(), 3]), unit = unit.nanometers)
        whether to input a unit.Quantity of conformers and bypass the conformer_generation/pymbar stage; None will default conduct this phase

    Returns
    -------
    conformers : unit.Quantity(np.array([num_iterations, system.getNumParticles(), 3]), unit = unit.nanometers)
        decorrelated positions of the md run
    rj_works : list
        work from each conformer proposal
    """
    from openmmtools import integrators
    from openmoltools.openeye import smiles_to_oemol, generate_conformers
    import simtk.unit as unit
    import simtk.openmm as openmm
    from openmmtools.constants import kB
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    import tqdm

    temperature = 300.0 * unit.kelvin # unit-bearing temperature
    kT = kB * temperature # unit-bearing thermal energy
    beta = 1.0/kT # unit-bearing inverse thermal energy

    #first, we must extract the top_prop relevant quantities
    system, topology = top_prop._old_system, top_prop._old_topology

    if prespecified_conformers == None:

        #now we can specify conformations from MD
        integrator = integrators.LangevinIntegrator(collision_rate = 1.0/unit.picosecond, timestep = 4.0*unit.femtosecond, temperature = temperature)
        context = openmm.Context(system, integrator)
        context.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(context)
        minimized_positions = context.getState(getPositions = True).getPositions(asNumpy = True)
        print(f"completed initial minimization")
        context.setPositions(minimized_positions)

        zeros = np.zeros([num_iterations, int(system.getNumParticles()), 3])
        conformers = unit.Quantity(zeros, unit=unit.nanometers)
        rps = np.zeros((num_iterations))

        print(f"conducting md sampling")
        for iteration in tqdm.trange(num_iterations):
            integrator.step(md_steps)
            state = context.getState(getPositions = True, getEnergy = True)
            new_positions = state.getPositions(asNumpy = True)
            conformers[iteration,:,:] = new_positions

            rp = state.getPotentialEnergy()*beta
            rps[iteration] = rp

        del context, integrator

        if compute_timeseries:
            print(f"computing production and data correlation")
            from pymbar import timeseries
            t0, g, Neff = timeseries.detectEquilibration(rps)
            series = timeseries.subsampleCorrelatedData(np.arange(t0, num_iterations), g = g)
            print(f"production starts at index {t0} of {num_iterations}")
            print(f"the number of effective samples is {Neff}")
            indices = t0 + series
            print(f"the filtered indices are {indices}")

        else:
            indices = range(num_iterations)
    else:
        conformers = prespecified_conformers
        indices = range(len(conformers))

    #now we can define a geometry_engine
    if geometry_method == 0:
        geometry_engine = FFAllAngleGeometryEngine( metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = True)
    elif geometry_method == 1:
        geometry_engine = FFAllAngleGeometryEngine( metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = False)
    elif geometry_method == 2:
        geometry_engine = FFAllAngleGeometryEngine( metadata=None, use_sterics=True, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = False)
    else:
        raise Exception(f"there is no geometry method for {geometry_method}")

    rj_works = []
    print(f"conducting geometry proposals...")
    for indx in tqdm.trange(len(indices)):
        index = indices[indx]
        print(f"index {indx}")
        new_positions, logp_forward = geometry_engine.propose(top_prop, conformers[index], beta)
        logp_backward = geometry_engine.logp_reverse(top_prop, new_positions, conformers[index], beta)
        print(f"\tlogp_forward, logp_backward: {logp_forward}, {logp_backward}")
        added_energy = geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential
        subtracted_energy = geometry_engine.reverse_final_context_reduced_potential - geometry_engine.reverse_atoms_with_positions_reduced_potential
        print(f"\tadded_energy, subtracted_energy: {added_energy}, {subtracted_energy}")
        work = logp_forward - logp_backward + added_energy - subtracted_energy
        rj_works.append(work)
        print(f"\ttotal work: {work}")

    return conformers, rj_works

def validate_endstate_energies(topology_proposal, htf, added_energy, subtracted_energy, beta = 1.0/kT, ENERGY_THRESHOLD = 1e-1):
    """
    Function to validate that the difference between the nonalchemical versus alchemical state at lambda = 0,1 is
    equal to the difference in valence energy (forward and reverse).

    Parameters
    ----------
    topology_proposal : perses.topology_proposal.TopologyProposal object
        top_proposal for relevant transformation
    htf : perses.new_relative.HybridTopologyFactory object
        hybrid top factory for setting alchemical hybrid states
    added_energy : float
        reduced added valence energy
    subtracted_energy: float
        reduced subtracted valence energy

    Returns
    -------
    zero_state_energy_difference : float
        reduced potential difference of the nonalchemical and alchemical lambda = 0 state (corrected for valence energy).
    one_state_energy_difference : float
        reduced potential difference of the nonalchemical and alchemical lambda = 1 state (corrected for valence energy).
    """
    import copy

    #create copies of old/new systems and set the dispersion correction
    top_proposal = copy.deepcopy(topology_proposal)
    top_proposal._old_system.getForce(3).setUseDispersionCorrection(False)
    top_proposal._new_system.getForce(3).setUseDispersionCorrection(False)

    #create copy of hybrid system, define old and new positions, and turn off dispersion correction
    hybrid_system = copy.deepcopy(htf.hybrid_system)
    hybrid_system_n_forces = hybrid_system.getNumForces()
    for force_index in range(hybrid_system_n_forces):
        forcename = hybrid_system.getForce(force_index).__class__.__name__
        if forcename == 'NonbondedForce':
            hybrid_system.getForce(force_index).setUseDispersionCorrection(False)

    old_positions, new_positions = htf._old_positions, htf._new_positions

    #generate endpoint thermostates
    nonalch_zero, nonalch_one, alch_zero, alch_one = generate_endpoint_thermodynamic_states(hybrid_system, top_proposal)

    # compute reduced energies
    #for the nonalchemical systems...
    attrib_list = [(nonalch_zero, old_positions, top_proposal._old_system.getDefaultPeriodicBoxVectors()),
                    (alch_zero, htf._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    (alch_one, htf._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    (nonalch_one, new_positions, top_proposal._new_system.getDefaultPeriodicBoxVectors())]

    rp_list = []
    platform = openmm.Platform.getPlatformByName('Reference')
    for (state, pos, box_vectors) in attrib_list:
        #print("\t\t\t{}".format(state))
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = state.create_context(integrator, platform)
        samplerstate = states.SamplerState(positions = pos, box_vectors = box_vectors)
        samplerstate.apply_to_context(context)
        rp = state.reduced_potential(context)
        rp_list.append(rp)
        energy_comps = compute_potential_components(context)
        for name, force in energy_comps:
           print("\t\t\t{}: {}".format(name, force))
        print(f'added forces:{sum([energy*beta for name, energy in energy_comps])}')
        print(f'rp: {rp}')
        del context, integrator

    #print(f"added_energy: {added_energy}; subtracted_energy: {subtracted_energy}")
    nonalch_zero_rp, alch_zero_rp, alch_one_rp, nonalch_one_rp = rp_list[0], rp_list[1], rp_list[2], rp_list[3]
    assert abs(nonalch_zero_rp - alch_zero_rp + added_energy) < ENERGY_THRESHOLD, f"The zero state alchemical and nonalchemical energy absolute difference {abs(nonalch_zero_rp - alch_zero_rp + added_energy)} is greater than the threshold of {ENERGY_THRESHOLD}."
    assert abs(nonalch_one_rp - alch_one_rp + subtracted_energy) < ENERGY_THRESHOLD, f"The one state alchemical and nonalchemical energy absolute difference {abs(nonalch_one_rp - alch_one_rp + subtracted_energy)} is greater than the threshold of {ENERGY_THRESHOLD}."


    return abs(nonalch_zero_rp - alch_zero_rp + added_energy), abs(nonalch_one_rp - alch_one_rp + subtracted_energy)
