import argparse
import json
import os
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    # Load model and vocabulary
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
    
    # Convert to undirected graph for visualization
    if isinstance(tree, nx.DiGraph):
        G = nx.Graph(tree)
    else:
        G = nx.Graph(tree)
    
    # Create node labels
    node_labels = {}
    for node in G.nodes():
        if isinstance(node, int) and node >= 0 and node < len(vocab):
            node_labels[node] = vocab[node]
        else:
            node_labels[node] = f"Internal {node}"
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, "tree_structure.png"))
    plt.show()
