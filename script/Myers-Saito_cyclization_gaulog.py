import numpy as np
import torch
import re
import os
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import math
from xyz2graph import MolGraph, to_networkx_graph
import dbstep.Dbstep as db
import sys


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding
def get_energy(path,i):
    with open(path, 'r') as file:
        lines = file.readlines()
        second_line = lines[1].strip()
    return (second_line.split('_')[i])

# Class for handling file operations
class FileHandler:
    @staticmethod
    def read_xyz(filename):
        with open(filename, 'r', encoding="utf-8") as file:
            num_atoms = int(file.readline().strip())
            comment = file.readline().strip()
            elements = []
            coordinates = []

            for line in file:
                parts = line.split()
                elements.append(parts[0])
                coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

        return num_atoms, comment, np.array(elements), np.array(coordinates)

    @staticmethod
    def format_xyz(num_atoms, comment, elements, coordinates):
        width = 15
        precision = 6
        lines = []
        lines.append(str(num_atoms))
        lines.append(comment)
        line_format = f"{{}} {{:>{width}.{precision}f}} {{:>{width}.{precision}f}} {{:>{width}.{precision}f}}"

        for element, coord in zip(elements, coordinates):
            line = line_format.format(element, coord[0], coord[1], coord[2])
            lines.append(line)

        return '\n'.join(lines)
    @staticmethod
    def delete_tmp_xyz():
        filename = 'tmp.xyz'
        if os.path.exists(filename):
            os.remove(filename)
# Class for extracting information from logs and descriptors
class DescriptorExtractor:
    @staticmethod
    def get_CDFT_Atom_descriptor(log_path: str) -> np.ndarray:
        """
        Extracts CDFT atom descriptors from a log file.

        Parameters:
            log_path (str): Path to the log file.

        Returns:
            np.ndarray: A 2D array containing CDFT atom descriptors.
        
        Raises:
            ValueError: If any of the required patterns are not found in the log file 
                        or if the descriptor values cannot be parsed.
        """
        with open(log_path, 'r') as file:
            text = file.read()

        # Clean the text
        text = re.sub(r'\s\)', ')', text)
        text = re.sub(r'[ \t]+', ' ', text)

        patterns = {
            'q_N': r'Atom q\(N\)(.*?)Condensed local electrophilicity',
            'electrophilicity': r'Atom\s+Electrophilicity(.*?)Condensed local softness',
            's_minus': r'Atom\s+s\-(.*?)E\(N\)',
        }

        matched_texts = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                raise ValueError(f"Pattern '{pattern}' not found in the log file.")
            # Split into lines and exclude the header and last two lines
            lines = match.group(1).strip().split('\n')[1:]
            matched_texts[key] = lines

        q_N = np.array([list(map(float, line.split()[1:])) for line in matched_texts['q_N']])
        electrophilicity = np.array([list(map(float, line.split()[1:])) for line in matched_texts['electrophilicity']])
        s_minus = np.array([list(map(float, line.split()[1:])) for line in matched_texts['s_minus']])

        # Concatenate all descriptors horizontally
        CDFT_descriptor = np.hstack((q_N, electrophilicity, s_minus))

        return np.array(CDFT_descriptor)

    @staticmethod
    def get_CDFT_Mol_descriptor(log_path: str) -> list:
        """
        Extracts CDFT molecular descriptors from a log file.

        Parameters:
            CDFT_path (str): Path to the CDFT log file.

        Returns:
            List[float]: A list of molecular descriptor values in eV.
        
        Raises:
            ValueError: If no molecular descriptors are found in the log file 
                        or if the descriptor values cannot be converted to float.
        """
        with open(log_path, 'r') as file:
            text = file.read()

        pattern = (
            r"(E_HOMO\(N\)|E_HOMO\(N\+1\)|First vertical IP|First vertical EA|"
            r"Mulliken electronegativity|Chemical potential|Hardness \(=fundamental gap\)|"
            r"Electrophilicity index|Nucleophilicity index):\s*[-\d.]+ Hartree,\s*([-]?\d+\.\d+) eV"
        )

        matches = re.findall(pattern, text)

        if not matches:
            raise ValueError("No molecular descriptors found in the log file.")

        values = [float(match[1]) for match in matches]

        return values
    @staticmethod
    def get_Mayer_bond(log_path: str, num_atoms: int) -> np.ndarray:
        """
        Extracts Mayer bond orders from a log file.

        Parameters:
            log_path (str): Path to the log file.

        Returns:
            np.ndarray: A 2D array containing Mayer bond orders.

        Raises:
            ValueError: If the log file does not contain enough lines or 
                        if the number of atoms cannot be parsed.
        """
        with open(log_path, 'r') as file:
            text = file.read()

        # Clean the text
        text = re.sub(r'\s\)', ')', text)
        text = re.sub(r'[ \t]+', ' ', text)

        lines = text.split('\n')
        if len(lines) < 4:
            raise ValueError("Log file does not contain enough lines to extract Mayer bond orders.")

        bond_order_lines = lines[2:-1]

        N_atom = num_atoms

        bond = np.zeros((N_atom, 0))  # Initialize with zero columns
        n = 0
        total_lines = len(bond_order_lines)

        while n + N_atom + 1 <= total_lines:
            # Assuming the first line in each block is a header or separator, skip it
            block = bond_order_lines[n + 1:n + N_atom + 1]
            bond_tmp = np.array([list(map(float, line.split()[1:])) for line in block])
            bond = np.hstack((bond, bond_tmp.reshape(N_atom, -1)))
            n += N_atom + 1

        return np.array(bond)

# Class for feature extraction from atoms and bonds
class FeatureExtractor:

    @staticmethod
    def get_atom_features(atom):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """
        # define list of permitted atoms
        permitted_list_of_atoms =  ['C','N','O','S','F','Cl','H']
        # compute atom features
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        is_in_a_ring_enc = [int(atom.IsInRing())]
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        atomic_mass_scaled = [float((atom.GetMass()))]
        vdw_radius_scaled = [float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))]
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())))]
        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
        return np.array(atom_feature_vector)


    @staticmethod
    def get_bond_features(bond, use_stereochemistry=False):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """

        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
        if use_stereochemistry == True:
            stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc
        return np.array(bond_feature_vector)

    @staticmethod
    def distance_index(arr1, arr2):
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        diff_squared = np.square(arr1 - arr2)
        distance = np.sqrt(np.sum(diff_squared))
        return distance

# Class for handling molecular graph processing
class MolecularGraphProcessor:
    def __init__(self, xyz_path, cdft_path, bndmat_path, unrelated_smiles="O=O"):
        self.xyz_path = xyz_path
        self.cdft_path = cdft_path
        self.bndmat_path = bndmat_path
        self.unrelated_smiles = unrelated_smiles
        self.unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        self.n_node_features = len(FeatureExtractor.get_atom_features(self.unrelated_mol.GetAtomWithIdx(0)))
        self.n_edge_features = len(FeatureExtractor.get_bond_features(self.unrelated_mol.GetBondBetweenAtoms(0, 1)))
        
        self.mg = MolGraph()
        self.mg.read_xyz(self.xyz_path)
        self.G = to_networkx_graph(self.mg)
        self.num_atoms, self.comment, self.elements, self.coordinates = FileHandler.read_xyz(self.xyz_path)

    def get_G_Side_chain(self, G):
        """
        Returns the side chains at both ends of EDY.

        :param G: The standardized graph
        :return: Indices of the two side chains and the frame part
        """
        ring_type = False
        G_0 = G.copy()
        G_0.remove_node(0)
        G_0.remove_node(5)
        connected_components = list(nx.connected_components(G_0))
        if len(connected_components) == 2:
            ring_type = True
            for i in connected_components:
                if 2 not in i: Side_chain1, Side_chain2 = i, i
                if 2 in i: Frame_Part = i
        
        elif len(connected_components) == 3:
            for i in connected_components:
                if 6 in i: Side_chain1 = i
                if 7 in i: Side_chain2 = i
                if 2 in i: Frame_Part = i
        return list(Side_chain1), list(Side_chain2), list(Frame_Part), ring_type

    def calculate_volume(self, elements, coordinates, Side_chain):
        tmp = FileHandler.format_xyz(len(Side_chain),"xyz",elements[Side_chain],coordinates[Side_chain])
        FileHandler.delete_tmp_xyz()
        with open('./tmp.xyz', 'w') as file:
            file.write(tmp)
        
        mol = db.dbstep("tmp.xyz",commandline=True,verbose=False,volume=True,quiet=True,measure='classic')  
        FileHandler.delete_tmp_xyz()
        return mol.bur_vol,mol.occ_vol


    def process(self):
        # Get Side Chains and Frame Part
        side_chain_1, side_chain_2, frame_part, ring_type = self.get_G_Side_chain(self.G)
        bur_vol_1, occ_vol_1 = self.calculate_volume(self.elements, self.coordinates, side_chain_1)
        bur_vol_2, occ_vol_2 = self.calculate_volume(self.elements, self.coordinates, side_chain_2)

        # Obtain global molecular descriptors
        global_features = [bur_vol_1, occ_vol_1, bur_vol_2, occ_vol_2] + DescriptorExtractor.get_CDFT_Mol_descriptor(self.cdft_path)

        # Load the molecule from the XYZ file and determine bonds
        raw_mol = Chem.MolFromXYZFile(self.xyz_path)
        mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineBonds(mol, charge=0)

        n_nodes = len(self.G.nodes)
        n_edges = 2 * len(self.G.edges)

        # Get atom-level CDFT descriptors
        atom_CDFT = DescriptorExtractor.get_CDFT_Atom_descriptor(self.cdft_path)

        # Construct the node feature matrix X
        X = np.zeros((n_nodes, self.n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = FeatureExtractor.get_atom_features(atom)

        # Construct distance feature matrix
        distance_feature = np.zeros((self.coordinates.shape[0], 8))
        reference_points = [
            [0, 0, 0],
            self.coordinates[0],
            self.coordinates[1],
            self.coordinates[2],
            self.coordinates[3],
            self.coordinates[4],
            self.coordinates[5],
            self.coordinates[6]
        ]

        for i in range(self.coordinates.shape[0]):
            for j, ref_point in enumerate(reference_points):
                distance_feature[i, j] = FeatureExtractor.distance_index(ref_point, self.coordinates[i])

        # Concatenate CDFT descriptors and distance features to node features
        X = np.hstack((atom_CDFT, distance_feature, X))

        # Get adjacency information for edges
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)

        # Construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, self.n_edge_features + 1))

        # Get bond order from Mayer bond matrix
        bond_order = DescriptorExtractor.get_Mayer_bond(self.bndmat_path, self.num_atoms)
        if bond_order.shape != (n_nodes, n_nodes):
            print("Error in bond order matrix dimensions")

        for k, (i, j) in enumerate(zip(rows, cols)):
            bond_features = FeatureExtractor.get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
            EF[k] = np.append(bond_features, bond_order[i, j])

        # Create tensors for model input
        y_val = float(get_energy(self.xyz_path,-1))
        # y_val = 0.0  # Placeholder value for target variable
        global_features_tensor = torch.tensor(global_features, dtype=torch.float)
        X_tensor = torch.tensor(X, dtype=torch.float)
        E_tensor = torch.stack([torch_rows, torch_cols], dim=0)
        EF_tensor = torch.tensor(EF, dtype=torch.float)
        y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)
        position_tensor = torch.tensor(self.coordinates, dtype=torch.float)

        # Output the tensors for further processing

        data = Data(
            x=X_tensor,
            edge_index=E_tensor,
            edge_attr=EF_tensor,
            y=y_tensor,
            global_features=global_features_tensor,
            pos=position_tensor,
            path=self.xyz_path,
            fragment=frame_part
        )
        torch.save(data, 'x0d_x1d.pt')
        print(data)

if __name__ == "__main__":
    processor = MolecularGraphProcessor(xyz_path=sys.argv[1], cdft_path="CDFT.txt", bndmat_path="bndmat.txt")
    processor.process()
