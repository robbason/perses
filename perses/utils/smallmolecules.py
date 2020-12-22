"""

Utility functions for handing small molecules

"""

__author__ = 'John D. Chodera'

import numpy as np
from openforcefield.topology import Molecule
from openforcefield.utils import get_data_file_path


def sdf_to_mols(sdf_filename, allow_undefined_stereo=False):
    # TODO: should this add hydrogens or generate conformers?
    mols_from_file = Molecule.from_file(
        sdf_filename,
        allow_undefined_stereo=allow_undefined_stereo)
    if isinstance(mols_from_file, Molecule):
        mols_from_file = [mols_from_file]
    return mols_from_file


def createMolFromSDF(sdf_filename, index=0, add_hydrogens=True, allow_undefined_stereo=False):
    """
    # Load an SDF file into an Mol. Since SDF files can contain multiple
    molecules, an index can be provided as well.

    Parameters
    ----------
    sdf_filename : str
        The name of the SDF file
    index : int, default 0
        The index of the molecule in the SDF file
    allow_undefined_stereo : bool, default=False
        wether to skip stereo perception

    Returns
    -------
    mol : openforcefield.topology.Molecule object
        The loaded mol object
    """

    mols_from_file =sdf_to_mols(
        sdf_filename,
        allow_undefined_stereo=allow_undefined_stereo)
    molecule = mols_from_file[index]
    assign_names_if_needed(molecule)
    molecule.generate_conformers()  # TODO: keep this?

    return molecule

def smiles_to_mol(smiles, title='MOL', max_confs=1):
    """
    Generate a mol from a SMILES string

    Parameters
    ----------
    smiles : str
        SMILES string of molecule
    title : str, default 'MOL'
        title of Mol molecule
    max_confs : int, default 1
        maximum number of conformers to generate
    Returns
    -------
    molecule : openforcefield.topology.Molecule
        Mol object of the molecule
    """

    # Create molecule
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=False)
    assign_names_if_needed(molecule)

    molecule.name = title

    # Assign geometry
    molecule.generate_conformers(n_conformers=max_confs)

    return molecule


def assign_names_if_needed(molecule):
    if not molecule.has_unique_atom_names:
        molecule.generate_unique_atom_names()


def describe_mol(mol):
    """
    Render the contents of an Mol to a string.

    Parameters
    ----------
    mol : Mol
        Molecule to describe

    Returns
    -------
    description : str
        The description
    """
    #TODO this needs a test
    description = ""
    description += "ATOMS:\n"
    for atom in mol.atoms:
        description += "%8d %5s %5d\n" % (atom.GetIdx(), atom.GetName(), atom.GetAtomicNum())
    description += "BONDS:\n"
    for bond in mol.bonds:
        description += "%8d %8d\n" % (bond.GetBgnIdx(), bond.GetEndIdx())
    return description


def sanitizeSMILES(smiles_list, mode='drop', verbose=False):
    """
    Sanitize set of SMILES strings by ensuring all are canonical isomeric SMILES.
    Duplicates are also removed.

    Parameters
    ----------
    smiles_list : iterable of str
        The set of SMILES strings to sanitize.
    mode : str, optional, default='drop'
        When a SMILES string that does not correspond to canonical isomeric SMILES is found, select the action to be performed.
        'exception' : raise an `Exception`
        'drop' : drop the SMILES string
        'expand' : expand all stereocenters into multiple molecules
    verbose : bool, optional, default=False
        If True, print verbose output.

    Returns
    -------
    sanitized_smiles_list : list of str
         Sanitized list of canonical isomeric SMILES strings.

    Examples
    --------
    Sanitize a simple list.
    >>> smiles_list = ['CC', 'CCC', '[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]']
    Throw an exception if undefined stereochemistry is present.
    >>> sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='exception')
    Traceback (most recent call last):
      ...
    Exception: Molecule '[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]' has undefined stereocenters
    Drop molecules iwth undefined stereochemistry.
    >>> sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='drop')
    >>> len(sanitized_smiles_list)
    2
    Expand molecules iwth undefined stereochemistry.
    >>> sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='expand')
    >>> len(sanitized_smiles_list)
    4
    """
    sanitized_smiles_set = set()
    #OESMILES_OPTIONS = oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_ISOMERIC | oechem.OESMILESFlag_Hydrogens  ## IVY
    for smiles in smiles_list:
        molecule = OEGraphMol()
        OESmilesToMol(molecule, smiles)

        oechem.OEAddExplicitHydrogens(molecule)

        if verbose:
            molecule.SetTitle(smiles)
            oechem.OETriposAtomNames(molecule)

        if has_undefined_stereocenters(molecule, verbose=verbose):
            if mode == 'drop':
                if verbose:
                    print("Dropping '%s' due to undefined stereocenters." % smiles)
                continue
            elif mode == 'exception':
                raise Exception("Molecule '%s' has undefined stereocenters" % smiles)
            elif mode == 'expand':
                if verbose:
                    print('Expanding stereochemistry:')
                    print('original: %s', smiles)
                molecules = enumerate_undefined_stereocenters(molecule, verbose=verbose)
                for molecule in molecules:
                    smiles_string = oechem.OECreateSmiString(molecule, OESMILES_OPTIONS)  ## IVY
                    sanitized_smiles_set.add(smiles_string)  ## IVY
                    if verbose: print('expanded: %s', smiles_string)
        else:
            # Convert to OpenEye's canonical isomeric SMILES.
            smiles_string = oechem.OECreateSmiString(molecule, OESMILES_OPTIONS) ## IVY
            sanitized_smiles_set.add(smiles_string) ## IVY

    sanitized_smiles_list = list(sanitized_smiles_set)

    return sanitized_smiles_list


def canonicalize_SMILES(smiles_list):
    """Ensure all SMILES strings end up in canonical form.
    Stereochemistry must already have been expanded.
    SMILES strings are converted to an OpenFF molecule and back again.
    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings
    Returns
    -------
    canonical_smiles_list : list of str
        List of SMILES strings, after canonicalization.
    """

    # Round-trip each molecule through a Molecule to end up in canonical form
    return [Molecule.from_smiles(smiles).to_smiles() for smiles in smiles_list]


def subset_molecule(molecule, included_atom_indexes):
    # TODO: should this capability go into openff, which would then be able to
    # call openeye's logic instead of this logic?
    # TODO: use RDKit's Murcko scaffold instead?
    # https://www.rdkit.org/docs/GettingStartedInPython.html#murcko-decomposition
    pred = set(included_atom_indexes)
    dst = Molecule()
    new_atoms = {}
    for atom in molecule.atoms:
        picked = atom.molecule_atom_index in pred
        if picked:
            new_atom = dst.add_atom(
                atom.atomic_number,
                atom.formal_charge,
                atom.is_aromatic, # TODO: can this change?
                name=atom.name)
            new_atoms[atom.molecule_atom_index] = new_atom

    for bond in molecule.bonds:
        idx1 = bond.atom1.molecule_atom_index
        idx2 = bond.atom2.molecule_atom_index
        if {idx1, idx2} < pred:
            dst.add_bond(
                new_atoms[idx1], new_atoms[idx2],
                bond.bond_order,
                bond.is_aromatic, # TODO: can this change?
                fractional_bond_order=bond.fractional_bond_order)
            # TODO: verify fractional_bond_order doesn't change
    return dst

def show_topology(topology):
    """
    Outputs bond atoms and bonds in topology object

    Paramters
    ----------
    topology : Topology object
    """
    output = ""
    for atom in topology.atoms():
        output += "%8d %5s %5s %3s: bonds " % (atom.index, atom.name, atom.residue.id, atom.residue.name)
        for bond in atom.residue.bonds():
            if bond[0] == atom:
                output += " %8d" % bond[1].index
            if bond[1] == atom:
                output += " %8d" % bond[0].index
        output += '\n'
    print(output)


def render_atom_mapping(filename, molecule1, molecule2,
                        new_to_old_atom_map, width=1200, height=600):
    """
    Render the atom mapping to a PDF file.

    Parameters
    ----------
    filename : str
        The PDF filename to write to.
    molecule1 : Molecule
        Initial molecule
    molecule2 : Molecule
        Final molecule
    new_to_old_atom_map : dict of int
        new_to_old_atom_map[molecule2_atom_index] is the corresponding molecule1 atom index
    width : int, optional, default=1200
        Width in pixels
    height : int, optional, default=1200
        Height in pixels

    """

    # Make copies of the input molecules so we can save the 2D coordinates
    # making a copy resets the atom indices, so the new_to_old_atom_map has to be remapped with the new, zero-indexed indices
    def indices(mol):
        return [atom.molecule_atom_index for atom in mol.atoms]
    # TODO: is openff's remap relevant here?
    molecule1_indices = indices(molecule1)
    molecule2_indices = indices(molecule2)
    molecule1 = Molecule(molecule1)
    molecule2 = Molecule(molecule2)
    molecule1_indices_new = indices(molecule1)
    molecule2_indices_new = indices(molecule2)

    modified_map_1 = {old: new for new, old
                      in zip(molecule1_indices_new, molecule1_indices)}
    modified_map_2 = {old: new for new, old
                      in zip(molecule2_indices_new, molecule2_indices)}
    new_to_old_atom_map = {modified_map_2[key]: modified_map_1[val]
                           for key, val in new_to_old_atom_map.items()}

    oechem.OEGenerate2DCoordinates(molecule1)
    oechem.OEGenerate2DCoordinates(molecule2)

    # Add both to an OEGraphMol reaction
    rmol = oechem.OEGraphMol()
    rmol.SetRxn(True)
    def add_molecule(mol):
        # Add atoms
        new_atoms = list()
        old_to_new_atoms = dict()
        for old_atom in mol.GetAtoms():
            new_atom = rmol.NewAtom(old_atom.GetAtomicNum())
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
            new_atoms.append(new_atom)
            old_to_new_atoms[old_atom] = new_atom
        # Add bonds
        for old_bond in mol.GetBonds():
            rmol.NewBond(old_to_new_atoms[old_bond.GetBgn()], old_to_new_atoms[old_bond.GetEnd()], old_bond.GetOrder())
        return new_atoms, old_to_new_atoms

    [new_atoms_1, old_to_new_atoms_1] = add_molecule(molecule1)
    [new_atoms_2, old_to_new_atoms_2] = add_molecule(molecule2)

    # Label reactant and product
    for atom in new_atoms_1:
        atom.SetRxnRole(oechem.OERxnRole_Reactant)
    for atom in new_atoms_2:
        atom.SetRxnRole(oechem.OERxnRole_Product)

    core1 = oechem.OEAtomBondSet()
    core2 = oechem.OEAtomBondSet()
    # add all atoms to the set
    core1.AddAtoms(new_atoms_1)
    core2.AddAtoms(new_atoms_2)
    # Label mapped atoms
    core_change = oechem.OEAtomBondSet()
    index =1
    for (index2, index1) in new_to_old_atom_map.items():
        new_atoms_1[index1].SetMapIdx(index)
        new_atoms_2[index2].SetMapIdx(index)
        # now remove the atoms that are core, so only uniques are highlighted
        core1.RemoveAtom(new_atoms_1[index1])
        core2.RemoveAtom(new_atoms_2[index2])
        if new_atoms_1[index1].GetAtomicNum() != new_atoms_2[index2].GetAtomicNum():
            # this means the element type is changing
            core_change.AddAtom(new_atoms_1[index1])
            core_change.AddAtom(new_atoms_2[index2])
        index += 1
    # Set up image options
    itf = oechem.OEInterface()
    oedepict.OEConfigureImageOptions(itf)
    ext = oechem.OEGetFileExtension(filename)
    if not oedepict.OEIsRegisteredImageFile(ext):
        raise Exception('Unknown image type for filename %s' % filename)
    ofs = oechem.oeofstream()
    if not ofs.open(filename):
        raise Exception('Cannot open output file %s' % filename)

    # Setup depiction options
    oedepict.OEConfigure2DMolDisplayOptions(itf, oedepict.OE2DMolDisplaySetup_AromaticStyle)
    opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
    oedepict.OESetup2DMolDisplayOptions(opts, itf)
    opts.SetBondWidthScaling(True)
    opts.SetAtomPropertyFunctor(oedepict.OEDisplayAtomMapIdx())
    opts.SetAtomColorStyle(oedepict.OEAtomColorStyle_WhiteMonochrome)

    # Depict reaction with component highlights
    oechem.OEGenerate2DCoordinates(rmol)
    rdisp = oedepict.OE2DMolDisplay(rmol, opts)

    if core1.NumAtoms() != 0:
        oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEPink),oedepict.OEHighlightStyle_Stick, core1)
    if core2.NumAtoms() != 0:
        oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEPurple),oedepict.OEHighlightStyle_Stick, core2)
    if core_change.NumAtoms() != 0:
        oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEGreen),oedepict.OEHighlightStyle_Stick, core_change)
    oedepict.OERenderMolecule(ofs, ext, rdisp)
    ofs.close()


def render_protein_residue_atom_mapping(topology_proposal, filename, width = 1200, height=600):
    """
    wrap the `render_atom_mapping` method around protein point mutation topologies.
    TODO : make modification to `render_atom_mapping` so that the backbone atoms are not written in the output.

    arguments
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal object
            topology proposal of protein mutation
        filename : str
            filename to write the map
        width : int
            width of image
        height : int
            height of image
    """
    from perses.utils.smallmolecules import render_atom_mapping
    oe_res_maps = {}
    for omm_new_idx, omm_old_idx in topology_proposal._new_to_old_atom_map.items():
        if omm_new_idx in topology_proposal._new_topology.residue_to_oemol_map.keys():
            try:
                oe_res_maps[topology_proposal._new_topology.residue_to_oemol_map[omm_new_idx]] = topology_proposal._old_topology.residue_to_oemol_map[omm_old_idx]
            except:
                pass

    render_atom_mapping(filename, topology_proposal._old_topology.residue_oemol, topology_proposal._new_topology.residue_oemol, oe_res_maps)


def generate_ligands_figure(molecules,figsize=None,filename='ligands.png'):
    """ Plot an image with all of the ligands passed in

    Parameters
    ----------
    molecules : list
        list of openeye.oemol objects
    figsize : list or tuple
        list or tuple of len() == 2 of the horizontal and vertical lengths of image
    filename : string
        name of file to save the image

    Returns
    -------

    """
    from openeye import oechem,oedepict

    to_draw = []
    for lig in molecules:
        oedepict.OEPrepareDepiction(lig)
        to_draw.append(oechem.OEGraphMol(lig))

    dim = int(np.ceil(len(to_draw)**0.5))

    if figsize is None:
        x_len = 1000*dim
        y_len = 500*dim
        image = oedepict.OEImage(x_len, y_len)
    else:
        assert ( len(figsize) == 2 ), "figsize arguement should be a tuple or list of length 2"
        image = oedepict.OEImage(figsize[0],figsize[1])

    rows, cols = dim, dim
    grid = oedepict.OEImageGrid(image, rows, cols)

    opts = oedepict.OE2DMolDisplayOptions(grid.GetCellWidth(), grid.GetCellHeight(), oedepict.OEScale_AutoScale)

    minscale = float("inf")
    for mol in to_draw:
        minscale = min(minscale, oedepict.OEGetMoleculeScale(mol, opts))
    #     print(mol.GetTitle())

    opts.SetScale(minscale)
    for idx, cell in enumerate(grid.GetCells()):
        mol = to_draw[idx]
        disp = oedepict.OE2DMolDisplay(mol, opts)
        oedepict.OERenderMolecule(cell, disp)

    oedepict.OEWriteImage(filename, image)

    return
