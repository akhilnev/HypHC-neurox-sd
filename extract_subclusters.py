import argparse
import json
import os
import numpy as np
import torch
import networkx as nx
from collections import defaultdict

def extract_subclusters(tree, vocab):
    """Extract subclusters from the hierarchical tree by finding branches."""
    # Convert to undirected graph if needed
    if isinstance(tree, nx.DiGraph):
        G = nx.Graph(tree)
    else:
        G = nx.Graph(tree)
    
    # Find the root node (typically has the highest degree)
    root = max(G.nodes(), key=lambda x: G.degree(x))
    print(f"Identified root node: {root} with degree {G.degree(root)}")
    
    # Find all branches from the root
    branches = []
    
    # Get immediate children of the root
    children = list(G.neighbors(root))
    print(f"Root has {len(children)} immediate children: {children}")
    
    # For each child, get its subtree
    for i, child in enumerate(children):
        # Remove the edge between root and child temporarily
        G.remove_edge(root, child)
        
        # Find the connected component containing this child
        subgraph_nodes = nx.node_connected_component(G, child)
        subgraph = G.subgraph(subgraph_nodes)
        
        # Add this branch
        branches.append(subgraph)
        
        # Print information about this branch
        print(f"Branch {i+1} has {len(subgraph_nodes)} nodes")
        
        # Restore the edge
        G.add_edge(root, child)
    
    # Extract tokens from each branch
    subclusters = []
    for i, branch in enumerate(branches):
        # Get leaf nodes in this branch
        leaves = [node for node in branch.nodes() if branch.degree(node) == 1]
        print(f"Branch {i+1} has {len(leaves)} leaf nodes")
        
        # Show some sample leaves
        sample_leaves = leaves[:5] if len(leaves) > 5 else leaves
        print(f"Sample leaves from branch {i+1}: {sample_leaves}")
        
        # Filter to valid token indices
        valid_leaves = [leaf for leaf in leaves if isinstance(leaf, int) and leaf >= 0 and leaf < len(vocab)]
        print(f"Branch {i+1} has {len(valid_leaves)} valid token indices")
        
        if valid_leaves:
            subclusters.append([vocab[leaf] for leaf in valid_leaves])
    
    return subclusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_size", type=int, default=1, help="Minimum cluster size to display")
    args = parser.parse_args()
    
    # Load configuration and model (same as before)
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    vocab = np.load("data/processed_activations/_processed_vocab.npy", allow_pickle=True)
    model_path = os.path.join(args.model_dir, f"model_{args.seed}.pkl")
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    from model.hyphc import HypHC
    model = HypHC(len(vocab), config.get("rank", 2), config.get("temperature", 0.1), 
                 config.get("init_size", 0.05), config.get("max_scale", 1.0))
    model.load_state_dict(model_state)
    
    # Decode tree
    tree = model.decode_tree(fast_decoding=True)
    
    # Extract subclusters
    subclusters = extract_subclusters(tree, vocab)
    
    # Print subclusters
    print(f"\nFound {len(subclusters)} subclusters:")
    for i, cluster in enumerate(subclusters):
        if len(cluster) >= args.min_size:
            print(f"\nSubcluster {i+1} ({len(cluster)} tokens):")
            for token in cluster:
                print(f"  {token}")
