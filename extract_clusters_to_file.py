import argparse
import json
import os
import numpy as np
import torch
import networkx as nx
import gc  # Garbage collection for memory management

def extract_clusters_to_file(model_dir, seed=0, output_file=None, min_size=2, max_tokens_per_cluster=100):
    """Extract clusters from model and save to a text file with memory optimizations"""
    print("Loading configuration and vocabulary...")
    
    # Load configuration
    config = json.load(open(os.path.join(model_dir, "config.json")))
    
    # Load vocabulary
    vocab = np.load("data/processed_activations/_processed_vocab.npy", allow_pickle=True)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(model_dir, f"model_{args.seed}.pkl")
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    
    from model.hyphc import HypHC
    model = HypHC(len(vocab), config.get("rank", 2), config.get("temperature", 0.1), 
                 config.get("init_size", 0.05), config.get("max_scale", 1.0))
    model.load_state_dict(model_state)
    
    # Decode tree
    print("Decoding tree (this may take a while)...")
    tree = model.decode_tree(fast_decoding=True)
    
    # Free memory
    del model_state
    del model
    gc.collect()
    
    # Print tree info
    print(f"Tree type: {type(tree)}")
    print(f"Number of nodes in tree: {len(tree.nodes())}")
    
    # Convert to undirected graph if needed
    if isinstance(tree, nx.DiGraph):
        G = nx.Graph(tree)
    else:
        G = nx.Graph(tree)
    
    # Find the root node (typically has the highest degree)
    root = max(G.nodes(), key=lambda x: G.degree(x))
    print(f"Identified root node: {root} with degree {G.degree(root)}")
    
    # Get output file path
    if output_file is None:
        output_file = os.path.join(model_dir, f"clusters_seed{seed}_min{min_size}.txt")
    
    print(f"Will write results to {output_file}")
    
    # Open output file
    with open(output_file, 'w') as f:
        f.write(f"Clusters from model {model_dir}, seed {seed}\n")
        f.write(f"Minimum cluster size: {min_size}\n\n")
        
        # Get immediate children of the root
        children = list(G.neighbors(root))
        print(f"Root has {len(children)} immediate children")
        
        # Process each branch to minimize memory usage
        total_clusters = 0
        
        for i, child in enumerate(children):
            print(f"Processing branch {i+1} of {len(children)}...")
            
            # Remove the edge between root and child temporarily
            G.remove_edge(root, child)
            
            # Find the connected component containing this child
            subgraph_nodes = nx.node_connected_component(G, child)
            
            # Get leaf nodes in this branch
            leaves = []
            for node in subgraph_nodes:
                if G.degree(node) == 1 and isinstance(node, int) and node >= 0 and node < len(vocab):
                    leaves.append(node)
            
            # Process if we have enough leaves
            if len(leaves) >= min_size:
                total_clusters += 1
                
                # Write cluster info to file
                f.write(f"Cluster {total_clusters} ({len(leaves)} tokens):\n")
                
                # Limit number of tokens to display if too many
                display_count = min(len(leaves), max_tokens_per_cluster)
                for j in range(display_count):
                    f.write(f"  {vocab[leaves[j]]}\n")
                
                if display_count < len(leaves):
                    f.write(f"  ... and {len(leaves) - display_count} more tokens\n")
                
                f.write("\n")
            
            # Restore the edge
            G.add_edge(root, child)
            
            # Free memory
            gc.collect()
            
            # Log progress periodically
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(children)} branches, found {total_clusters} clusters so far")
        
        # Write summary at the end
        f.write(f"\nTotal clusters found: {total_clusters}")
        
    print(f"Done! Found {total_clusters} clusters and wrote results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--seed", type=int, default=0, help="Model seed to use")
    parser.add_argument("--min_size", type=int, default=2, help="Minimum cluster size")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (default: {model_dir}/clusters_seed{seed}_min{min_size}.txt)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to display per cluster")
    
    args = parser.parse_args()
    
    extract_clusters_to_file(
        args.model_dir, 
        args.seed, 
        args.output_file, 
        args.min_size, 
        args.max_tokens
    )
