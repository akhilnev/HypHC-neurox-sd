import argparse
import json
import os
import numpy as np
import torch
import networkx as nx
from collections import defaultdict

#! Problem with this was we have one big connected component so subgraphs wont exist rather we get one big connected component / Cluster, so we need to split that

def extract_clusters(tree, vocab):
    """Extract clusters from the hierarchical tree."""
    # First, let's understand the tree structure
    print("Tree type:", type(tree))
    print("Tree structure sample:", str(tree)[:500] + "..." if len(str(tree)) > 500 else str(tree))
    
    # If tree is a directed graph, convert to undirected
    if isinstance(tree, nx.DiGraph):
        G = nx.Graph(tree)  # Convert directed graph to undirected
        print("Converted directed graph to undirected")
    elif isinstance(tree, nx.Graph):
        # If tree is already an undirected NetworkX graph
        G = tree
    elif isinstance(tree, list) and all(isinstance(item, tuple) for item in tree):
        # If tree is a list of tuples (edges)
        G = nx.Graph()
        for edge in tree:
            G.add_edge(edge[0], edge[1])
    elif isinstance(tree, dict):
        # If tree is a dictionary (adjacency list)
        G = nx.Graph()
        for node, neighbors in tree.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
    else:
        # Try to iterate through tree as edges
        try:
            G = nx.Graph()
            for edge in tree:
                if hasattr(edge, '__iter__') and len(edge) == 2:
                    G.add_edge(edge[0], edge[1])
                else:
                    print(f"Unexpected edge format: {edge}")
        except Exception as e:
            print(f"Error processing tree: {e}")
            print("Please check the model.decode_tree() implementation for the exact format")
            return []
    
    # Find all connected components (these are the clusters)
    clusters = list(nx.connected_components(G))
    
    # For each cluster, get the leaf nodes (these are your tokens)
    cluster_tokens = []
    for cluster in clusters:
        leaves = [node for node in cluster if G.degree(node) == 1]
        # Filter out internal nodes (they have negative indices in the HypHC implementation)
        leaves = [leaf for leaf in leaves if isinstance(leaf, int) and leaf >= 0 and leaf < len(vocab)]
        if leaves:  # Only include clusters with actual tokens
            cluster_tokens.append([vocab[leaf] for leaf in leaves])
    
    return cluster_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_size", type=int, default=1, help="Minimum cluster size to display")
    args = parser.parse_args()
    
    # Load configuration
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    
    # Load vocabulary
    vocab = np.load("data/processed_activations/_processed_vocab.npy", allow_pickle=True)
    
    # Load model
    model_path = os.path.join(args.model_dir, f"model_{args.seed}.pkl")
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load the model class
    from model.hyphc import HypHC
    model = HypHC(len(vocab), config.get("rank", 2), config.get("temperature", 0.1), 
                 config.get("init_size", 0.05), config.get("max_scale", 1.0))
    model.load_state_dict(model_state)
    
    # Decode tree
    tree = model.decode_tree(fast_decoding=True)
    
    # Inspect tree structure
    print("Tree type:", type(tree))
    print("Tree structure sample:", str(tree)[:500] + "..." if len(str(tree)) > 500 else str(tree))
    
    # If tree is a list, inspect the first few elements
    if isinstance(tree, list):
        print("\nFirst 5 elements:")
        for i, item in enumerate(tree[:5]):
            print(f"Element {i}: {type(item)} - {item}")
    
    # If tree is a dictionary, inspect some keys and values
    elif isinstance(tree, dict):
        print("\nSome keys and values:")
        for i, (key, value) in enumerate(list(tree.items())[:5]):
            print(f"Key: {key}, Value type: {type(value)}, Value: {value}")
    
    # Extract clusters
    clusters = extract_clusters(tree, vocab)
    
    # Print clusters
    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        if len(cluster) >= args.min_size:
            print(f"\nCluster {i+1} ({len(cluster)} tokens):")
            for token in cluster:
                print(f"  {token}")
