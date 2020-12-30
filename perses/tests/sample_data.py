from pkg_resources import resource_filename
from perses.utils.smallmolecules import sdf_to_mols


def load_JACS(dataset_name):
    dataset_path = f'data/schrodinger-jacs-datasets/{dataset_name}_ligands.sdf'
    sdf_filename = resource_filename('perses', dataset_path)
    return {mol.name: mol for mol in sdf_to_mols(sdf_filename)}


def load_simple():
    dataset_path = f'data/simple_mol/two.sdf'
    sdf_filename = resource_filename('perses', dataset_path)
    return {mol.name: mol for mol in sdf_to_mols(sdf_filename)}
