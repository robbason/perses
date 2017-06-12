from simtk.openmm import app
from simtk import unit, openmm
import numpy as np
from perses.annihilation.relative import HybridTopologyFactory
import copy

ace = {
    'H1'  : [app.Element.getBySymbol('H'), (2.022 ,  0.992 ,  0.038)],#  1.00  0.00           H
    'CH3' : [app.Element.getBySymbol('C'), (1.990 ,  2.080 ,  0.002)],#  1.00  0.00           C
    'H2'  : [app.Element.getBySymbol('H'), (1.506 ,  2.554 ,  0.838)],#  1.00  0.00           H
    'H3'  : [app.Element.getBySymbol('H'), (1.466 ,  2.585 , -0.821)],#  1.00  0.00           H
    'C'   : [app.Element.getBySymbol('C'), (3.423 ,  2.632 ,  0.023)],#  1.00  0.00           C
    'O'   : [app.Element.getBySymbol('O'), (4.402 ,  1.911 , -0.004)],#  1.00  0.00           O
}

nme = {
    'N'   : [app.Element.getBySymbol('N'), (5.852 ,  6.852 ,  0.008)],#  1.00  0.00           N
    'H'   : [app.Element.getBySymbol('H'), (6.718 ,  6.342 , -0.055)],#  1.00  0.00           H
    'C'   : [app.Element.getBySymbol('C'), (5.827 ,  8.281 ,  0.014)],#  1.00  0.00           C
    'H1'  : [app.Element.getBySymbol('H'), (4.816 ,  8.703 , -0.069)],#  1.00  0.00           H
    'H2'  : [app.Element.getBySymbol('H'), (6.407 ,  8.745 ,  0.826)],#  1.00  0.00           H
    'H3'  : [app.Element.getBySymbol('H'), (6.321 ,  8.679 , -0.867)],#  1.00  0.00           H
}

core = {
    'N'  : [app.Element.getBySymbol('N'), (3.547 ,  3.932 , -0.028)],#  1.00  0.00           N
    'H'  : [app.Element.getBySymbol('H'), (2.712 ,  4.492 , -0.088)],#  1.00  0.00           H
    'CA' : [app.Element.getBySymbol('C'), (4.879 ,  4.603 ,  0.004)],#  1.00  0.00           C
    'HA' : [app.Element.getBySymbol('H'), (5.388 ,  4.297 ,  0.907)],#  1.00  0.00           H
    'C'  : [app.Element.getBySymbol('C'), (4.724 ,  6.133 , -0.020)],
    'O'  : [app.Element.getBySymbol('O'), (3.581 ,  6.640 ,  0.027)]
}

ala_unique = {
    'CB'  : [app.Element.getBySymbol('C'), (5.665 ,  4.222 , -1.237)],#  1.00  0.00           C
    'HB1' : [app.Element.getBySymbol('H'), (5.150 ,  4.540 , -2.116)],#  1.00  0.00           H
    'HB2' : [app.Element.getBySymbol('H'), (6.634 ,  4.705 , -1.224)],#  1.00  0.00           H
    'HB3' : [app.Element.getBySymbol('H'), (5.865 ,  3.182 , -1.341)],#  1.00  0.00           H
}

leu_unique = {
    'CB'  : [app.Element.getBySymbol('C'), (5.840 ,  4.228 , -1.172)],#  1.00  0.00           C
    'HB2' : [app.Element.getBySymbol('H'), (5.192 ,  3.909 , -1.991)],#  1.00  0.00           H
    'HB3' : [app.Element.getBySymbol('H'), (6.549 ,  3.478 , -0.826)],#  1.00  0.00           H
    'CG'  : [app.Element.getBySymbol('C'), (6.398 ,  5.525 , -1.826)],#  1.00  0.00           C
    'HG'  : [app.Element.getBySymbol('H'), (6.723 ,  5.312 , -2.877)],#  1.00  0.00           H
    'CD1' : [app.Element.getBySymbol('C'), (7.770 ,  5.753 , -1.221)],#  1.00  0.00           C
    'HD11': [app.Element.getBySymbol('H'), (8.170 ,  6.593 , -1.813)],#  1.00  0.00           H
    'HD12': [app.Element.getBySymbol('H'), (8.420 ,  4.862 , -1.288)],#  1.00  0.00           H
    'HD13': [app.Element.getBySymbol('H'), (7.793 ,  5.788 , -0.123)],#  1.00  0.00           H
    'CD2' : [app.Element.getBySymbol('C'), (5.182 ,  6.334 , -2.328)],#  1.00  0.00           C
    'HD21': [app.Element.getBySymbol('H'), (5.460 ,  7.247 , -2.790)],#  1.00  0.00           H
    'HD22': [app.Element.getBySymbol('H'), (4.353 ,  5.833 , -2.769)],#  1.00  0.00           H
    'HD23': [app.Element.getBySymbol('H'), (4.798 ,  6.958 , -1.550)],#  1.00  0.00           H
}

core_bonds = [
    ('ace-C','ace-O'),
    ('ace-C','ace-CH3'),
    ('ace-CH3','ace-H1'),
    ('ace-CH3','ace-H2'),
    ('ace-CH3','ace-H3'),
    ('ace-C','N'),
    ('N', 'H'),
    ('N', 'CA'),
    ('CA', 'HA'),
    ('CA', 'C'),
    ('C', 'O'),
    ('C','nme-N'),
    ('nme-N','nme-H'),
    ('nme-N','nme-C'),
    ('nme-C','nme-H1'),
    ('nme-C','nme-H2'),
    ('nme-C','nme-H3'),
]

ala_bonds = [
    ('CA', 'CB'),
    ('CB', 'HB1'),
    ('CB', 'HB2'),
    ('CB', 'HB3')
]

leu_bonds = [
    ('CA', 'CB'),
    ('CB', 'HB2'),
    ('CB', 'HB3'),
    ('CB', 'CG'),
    ('CG', 'HG'),
    ('CG', 'CD1'),
    ('CG', 'CD2'),
    ('CD1', 'HD11'),
    ('CD1', 'HD12'),
    ('CD1', 'HD13'),
    ('CD2', 'HD21'),
    ('CD2', 'HD22'),
    ('CD2', 'HD23')
]

forcefield = app.ForceField('amber99sbildn.xml')

def get_available_parameters(system, prefix='lambda'):
    parameters = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if hasattr(force, 'getNumGlobalParameters'):
            for parameter_index in range(force.getNumGlobalParameters()):
                parameter_name = force.getGlobalParameterName(parameter_index)
                if parameter_name[0:(len(prefix)+1)] == (prefix + '_'):
                    parameters.append(parameter_name)
    return parameters

def build_two_residues():
    alanine_topology = app.Topology()
    alanine_positions = unit.Quantity(np.zeros((22,3)),unit.nanometer)
    leucine_topology = app.Topology()
    leucine_positions = unit.Quantity(np.zeros((31,3)),unit.nanometer)

    ala_chain = alanine_topology.addChain(id='A')
    leu_chain = leucine_topology.addChain(id='A')

    ala_ace = alanine_topology.addResidue('ACE', ala_chain)
    ala_res = alanine_topology.addResidue('ALA', ala_chain)
    ala_nme = alanine_topology.addResidue('NME', ala_chain)
    leu_ace = leucine_topology.addResidue('ACE', leu_chain)
    leu_res = leucine_topology.addResidue('LEU', leu_chain)
    leu_nme = leucine_topology.addResidue('NME', leu_chain)

    ala_atoms = dict()
    leu_atoms = dict()
    atom_map = dict()

    for core_atom_name, [element, position] in ace.items():
        position = np.asarray(position)
        alanine_positions[len(ala_atoms.keys())] = position*unit.angstrom
        leucine_positions[len(leu_atoms.keys())] = position*unit.angstrom

        ala_atom = alanine_topology.addAtom(core_atom_name, element, ala_ace)
        leu_atom = leucine_topology.addAtom(core_atom_name, element, leu_ace)

        ala_atoms['ace-'+core_atom_name] = ala_atom
        leu_atoms['ace-'+core_atom_name] = leu_atom
        atom_map[ala_atom.index] = leu_atom.index

    for core_atom_name, [element, position] in core.items():
        position = np.asarray(position)
        alanine_positions[len(ala_atoms.keys())] = position*unit.angstrom
        leucine_positions[len(leu_atoms.keys())] = position*unit.angstrom

        ala_atom = alanine_topology.addAtom(core_atom_name, element, ala_res)
        leu_atom = leucine_topology.addAtom(core_atom_name, element, leu_res)

        ala_atoms[core_atom_name] = ala_atom
        leu_atoms[core_atom_name] = leu_atom
        atom_map[ala_atom.index] = leu_atom.index

    for ala_atom_name, [element, position] in ala_unique.items():
        position = np.asarray(position)
        alanine_positions[len(ala_atoms.keys())] = position*unit.angstrom
        ala_atom = alanine_topology.addAtom(ala_atom_name, element, ala_res)
        ala_atoms[ala_atom_name] = ala_atom

    for leu_atom_name, [element, position] in leu_unique.items():
        position = np.asarray(position)
        leucine_positions[len(leu_atoms.keys())] = position*unit.angstrom
        leu_atom = leucine_topology.addAtom(leu_atom_name, element, leu_res)
        leu_atoms[leu_atom_name] = leu_atom

    for core_atom_name, [element, position] in nme.items():
        position = np.asarray(position)
        alanine_positions[len(ala_atoms.keys())] = position*unit.angstrom
        leucine_positions[len(leu_atoms.keys())] = position*unit.angstrom

        ala_atom = alanine_topology.addAtom(core_atom_name, element, ala_nme)
        leu_atom = leucine_topology.addAtom(core_atom_name, element, leu_nme)

        ala_atoms['nme-'+core_atom_name] = ala_atom
        leu_atoms['nme-'+core_atom_name] = leu_atom
        atom_map[ala_atom.index] = leu_atom.index

    for bond in core_bonds:
        alanine_topology.addBond(ala_atoms[bond[0]],ala_atoms[bond[1]])
        leucine_topology.addBond(leu_atoms[bond[0]],leu_atoms[bond[1]])
    for bond in ala_bonds:
        alanine_topology.addBond(ala_atoms[bond[0]],ala_atoms[bond[1]])
    for bond in leu_bonds:
        leucine_topology.addBond(leu_atoms[bond[0]],leu_atoms[bond[1]])

    return alanine_topology, alanine_positions, leucine_topology, leucine_positions, atom_map

def compute_alchemical_correction(unmodified_old_system, unmodified_new_system, alchemical_system, initial_positions, alchemical_positions, final_hybrid_positions, final_positions):

    def compute_logP(system, positions, parameter=None):
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = openmm.Context(system, integrator)
        context.setPositions(positions)
        context.applyConstraints(integrator.getConstraintTolerance())
        if parameter is not None:
            available_parameters = get_available_parameters(system)
            for parameter_name in available_parameters:
                context.setParameter(parameter_name, parameter)
        potential = context.getState(getEnergy=True).getPotentialEnergy()
        print('Potential: %s' % potential)
        del context, integrator
        return potential


    forces_to_save = {
        'Bond and Nonbonded' : ['HarmonicBondForce', 'CustomBondForce', 'NonbondedForce', 'CustomNonbondedForce'],
        'Angle' : ['HarmonicAngleForce', 'CustomAngleForce'],
        'Torsion' : ['PeriodicTorsionForce', 'CustomTorsionForce'],
        'CMMotion' : ['CMMotionRemover'],
        'All' : []
    }

    for saved_force, force_names in forces_to_save.items():
        print('\nPotential using %s Force:' % saved_force)
        unmodified_old_sys = copy.deepcopy(unmodified_old_system)
        unmodified_new_sys = copy.deepcopy(unmodified_new_system)
        alchemical_sys = copy.deepcopy(alchemical_system)
        for unmodified_system in [unmodified_old_sys, unmodified_new_sys, alchemical_sys]:
            if unmodified_system == alchemical_sys and saved_force == 'Bond and Nonbonded': max_forces = 5
            elif saved_force == 'Bond and Nonbonded': max_forces = 2
            elif saved_force == 'All': max_forces = unmodified_system.getNumForces() + 10
            else: max_forces = 1
            while unmodified_system.getNumForces() > max_forces:
                for k, force in enumerate(unmodified_system.getForces()):
                    force_name = force.__class__.__name__
                    if not force_name in force_names:
                        unmodified_system.removeForce(k)
                        break
        # Compute correction from transforming real system to/from alchemical system
        print('Inital, hybrid - physical')
        initial_logP_correction = compute_logP(alchemical_sys, alchemical_positions, parameter=0) - compute_logP(unmodified_old_sys, initial_positions)
        print('Final, physical - hybrid')
        final_logP_correction = compute_logP(unmodified_new_sys, final_positions) - compute_logP(alchemical_sys, final_hybrid_positions, parameter=1)
        print('Difference in Initial potentials:')
        print(initial_logP_correction)
        print('Difference in Final potentials:')
        print(final_logP_correction)
        logP_alchemical_correction = initial_logP_correction + final_logP_correction


def test_setup_hybrid_system():
    from topology_proposal import TopologyProposal
    import new_relative
    alanine_topology, alanine_positions, leucine_topology, leucine_positions, atom_map = build_two_residues()

    alanine_system = forcefield.createSystem(alanine_topology)
    leucine_system = forcefield.createSystem(leucine_topology)

    atom_map = {value : key for key, value in atom_map.items()}
    #hybrid = HybridTopologyFactory(leucine_system, alanine_system, leucine_topology, alanine_topology, leucine_positions, alanine_positions, atom_map, softening=0.0)
    top_prop = TopologyProposal(new_topology=leucine_topology, new_system=leucine_system, old_topology=alanine_topology, old_system=alanine_system, new_to_old_atom_map=atom_map)
    hybrid_factory = new_relative.HybridTopologyFactory(top_prop)

    [system, topology, positions, sys2_indices_in_system, sys1_indices_in_system] = hybrid.createPerturbedSystem()

    compute_alchemical_correction(leucine_system, alanine_system, hybrid_new, leucine_positions, positions, positions, alanine_positions)

def compare_energies(mol_name="naphthalene", ref_mol_name="benzene"):
    """
    Make an atom map where the molecule at either lambda endpoint is identical, and check that the energies are also the same.
    """
    from topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal
    import openeye.oechem as oechem
    from new_relative import HybridTopologyFactory
    import simtk.openmm as openmm

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC

    mol_name = "naphthalene"
    ref_mol_name = "benzene"

    mol = createOEMolFromIUPAC(mol_name)
    m, system, positions, topology = createSystemFromIUPAC(mol_name)

    refmol = createOEMolFromIUPAC(ref_mol_name)

    #map one of the rings
    atom_map = SmallMoleculeSetProposalEngine._get_mol_atom_map(mol, refmol)

    #now use the mapped atoms to generate a new and old system with identical atoms mapped. This will result in the
    #same molecule with the same positions for lambda=0 and 1, and ensures a contiguous atom map
    effective_atom_map = {value : value for value in atom_map.values()}

    #make a topology proposal with the appropriate data:
    top_proposal = TopologyProposal(new_topology=topology, new_system=system, old_topology=topology, old_system=system, new_to_old_atom_map=effective_atom_map, new_chemical_state_key="n1", old_chemical_state_key='n2')

    factory = HybridTopologyFactory(top_proposal, positions, positions)

    alchemical_system = factory.hybrid_system
    alchemical_positions = factory.hybrid_positions

    integrator = openmm.VerletIntegrator(1)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(alchemical_system, integrator, platform)

    context.setPositions(alchemical_positions)

    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : '1.0', # don't soften bonds
        'lambda_angles' : '1.0', # don't soften angles
        'lambda_torsions' : 'lambda'
    }

    #set all to zero
    for parm in functions.keys():
        context.setParameter(parm, 0.0)

    initial_energy = context.getState(getEnergy=True).getPotentialEnergy()

    #set all to one
    for parm in functions.keys():
        context.setParameter(parm, 1.0)

    final_energy = context.getState(getEnergy=True).getPotentialEnergy()

    if np.abs(final_energy - initial_energy) > 1.0e-6*unit.kilojoule_per_mole:
        raise Exception("The energy at the endpoints was not equal for molecule %s" % mol_name)

def test_compare_energies():
    mols_and_refs = [['naphthalene', 'benzene'], ['pentane', 'propane'], ['biphenyl', 'benzene']]

    for mol_ref_pair in mols_and_refs:
        compare_energies(mol_name=mol_ref_pair[0], ref_mol_name=mol_ref_pair[1])

if __name__ == '__main__':
    test_compare_energies()
