#!/usr/bin/env python
# coding: utf-8


import networkx as nx
from networkx.classes.graph import Graph
from itertools import combinations
import numpy as np
from collections import deque
import re
from typing import List
from xyz2graph import MolGraph, to_networkx_graph, to_plotly_figure

def read_xyz(filename):
    with open(filename, 'r',encoding="utf-8") as file:
        num_atoms = int(file.readline().strip())
        comment = file.readline().strip()
        elements = []
        coordinates = []

        for line in file:
            parts = line.split()
            elements.append(parts[0])
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return num_atoms, comment, np.array(elements), np.array(coordinates)

def format_xyz(num_atoms, comment, elements, coordinates):
    width=15
    precision=6
    lines = []
    lines.append(str(num_atoms))
    lines.append(comment)
    line_format = f"{{}} {{:>{width}.{precision}f}} {{:>{width}.{precision}f}} {{:>{width}.{precision}f}}"

    for element, coord in zip(elements, coordinates):
        line = line_format.format(element, coord[0], coord[1], coord[2])
        lines.append(line)

    return '\n'.join(lines)

def transform_coordinates(coords, centroid, transform_matrix):
    transformed_coords = coords - centroid 
    return np.dot(transformed_coords, transform_matrix)
def get_stand_transform_matrix(coordinates: np.ndarray,path_node: List[int]):
    
    selected_coordinates = coordinates[path_node]
    centroid = np.mean(selected_coordinates, axis=0)
    M = np.hstack((selected_coordinates, np.ones((7, 1))))
    U, S, Vt = np.linalg.svd(M)
    plane = Vt[-1, :]
    centroid = np.mean(selected_coordinates, axis=0)
    x_direction = np.mean(coordinates[path_node[2:4]], axis=0) - centroid
    z_direction = plane[:3]
    x_direction /= np.linalg.norm(x_direction)
    z_direction /= np.linalg.norm(z_direction)

    y_direction = np.cross(z_direction, x_direction)
    y_direction /= np.linalg.norm(y_direction)
    
    transform_matrix = np.column_stack((x_direction, y_direction, z_direction))
    
    stand_coordinates = []
    for i in coordinates:
        transformed_point = transform_coordinates(i, centroid, transform_matrix)
        stand_coordinates.append(transformed_point)
    stand_coordinates = np.array(stand_coordinates, dtype=np.float32)
    return stand_coordinates

def get_EDY_Subgraph(G: nx.classes.graph) -> np.ndarray:
    """
    Calculate the enediyne scaffold structure for the given molecular graph.
    
    Args:
        G (Graph): A networkx graph object representing the molecule.
        
    Returns:
        np.ndarray: An array of length  7 representing the enediyne scaffold.

    """
    valid_subgraphs = []
    nodes_with_label_C = [node for node, attr in G.nodes(data=True) if attr.get('label') == 'C']
    for nodes in combinations(nodes_with_label_C, 7):
        if is_valid_subgraph(G, nodes):
            valid_subgraphs.append(list(nodes))
    if len(valid_subgraphs) == 1:
        path_nodes = ordered_path_nodes(G,valid_subgraphs[0])
        return (np.array(path_nodes))
    else:
        for node in valid_subgraphs:
            path_nodes = ordered_path_nodes(G,node)
            if [G.degree[i] for i in path_nodes]==[2, 2, 3, 3, 3, 2, 3]:return (np.array(path_nodes))
        return valid_subgraphs

def ordered_path_nodes(G: nx.classes.graph, subgraph_nodes):

    # Create subgraph
    subgraph = G.subgraph(subgraph_nodes).copy()

    start_node = next((node for node in subgraph.nodes() if subgraph.degree(node) == 1 and G.degree(node) == 2), None)

    # If a starting node is found, perform DFS traversal
    if start_node is not None:
        return list(nx.dfs_preorder_nodes(subgraph, source=start_node))
    else:
        return []  # Return an empty list if no degree-1 node is found

def bfs_relabel(G: Graph, start_nodes: List[int]):
    visited = set()  # Record visited nodes
    queue = deque(start_nodes)  # Use a queue for BFS
    mapping = {}  # Old node to new node mapping
    new_label = 0  # New node label

    for node in start_nodes:
        mapping[node] = new_label
        new_label += 1

    while queue:
        node = queue.popleft()
        visited.add(node)
        # Add all adjacent unvisited nodes to the queue
        for neighbor in G.neighbors(node):
            if neighbor not in visited and neighbor not in mapping:
                mapping[neighbor] = new_label
                new_label += 1
                queue.append(neighbor)

    return mapping

def is_valid_subgraph(G: Graph, nodes) -> bool:
    subgraph = G.subgraph(nodes)
    if not nx.is_tree(subgraph):
        return False
        
    max_degree = max(dict(subgraph.degree()).values())
    if max_degree > 2:
        return False

    degree_total=[]
    for node in nodes:
        degree_total.append(G.degree(node))
    degree_total.sort()
    if degree_total == [2, 2, 2, 3, 3, 3, 3]:
        return  True
        
    return False


import sys
args = sys.argv

def work_std(XYZ_path):


    mg = MolGraph()
    mg.read_xyz(XYZ_path)
    G = to_networkx_graph(mg)
    num_atoms, comment, elements, coordinates = read_xyz(XYZ_path)

    for i in G.nodes():
        G.nodes[i]["label"] = elements[i]


    path_node = get_EDY_Subgraph(G)
    stand_coordinates = get_stand_transform_matrix(coordinates,path_node)

    mapping = bfs_relabel(G, path_node)
    if len(mapping)!=num_atoms:
        print("error")
    new_coordinates = np.empty_like(stand_coordinates)
    new_elements = np.empty_like(elements)

    for old_node, new_node in mapping.items():
        new_coordinates[new_node] = stand_coordinates[old_node]
        new_elements[new_node] = elements[old_node]
        

    xyz_content = format_xyz(num_atoms, comment, new_elements, new_coordinates)
    with open(XYZ_path.replace('.xyz','_std.xyz'), 'w') as file:
        file.write(xyz_content)  

import sys
work_std(sys.argv[1])

