"""Script to visualize the HypHC clustering."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import networkx as nx

from datasets.loading import load_data
from model.hyphc import HypHC
from utils.poincare import project
from utils.visualization import plot_tree_from_leaves

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="path to a directory with a torch model_{seed}.pkl and a config.json files saved by train.py."
                        )
    parser.add_argument("--seed", type=str, default=0, help="model seed to use")
    args = parser.parse_args()

    # load dataset
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    config_args = argparse.Namespace(**config)
    _, y_true, similarities = load_data(config_args.dataset)

    # build HypHC model
    model = HypHC(similarities.shape[0], config_args.rank, config_args.temperature, config_args.init_size,
                  config_args.max_scale)
    params = torch.load(os.path.join(args.model_dir, f"model_{args.seed}.pkl"), map_location=torch.device('cpu'))
    model.load_state_dict(params, strict=False)
    model.eval()

    # decode tree
    tree = model.decode_tree(fast_decoding=True)
    leaves_embeddings = model.normalize_embeddings(model.embeddings.weight.data)
    leaves_embeddings = project(leaves_embeddings).detach().cpu().numpy()

    # Create a more readable layout
    pos = nx.kamada_kawai_layout(tree)  # or nx.spring_layout(tree, k=0.5)

    # Use different colors for internal nodes vs. tokens
    node_colors = ['skyblue' if 'Internal' in str(tree.nodes[node]['label']) else 'orange' for node in tree.nodes()]

    # Adjust node sizes
    node_sizes = [300 if 'Internal' in str(tree.nodes[node]['label']) else 500 for node in tree.nodes()]

    # Draw with improved settings
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(tree, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(tree, pos, width=0.5, alpha=0.5)

    # Only show labels for actual tokens, not internal nodes
    token_labels = {node: label for node, label in tree.nodes(data='label') if not 'Internal' in str(label)}
    nx.draw_networkx_labels(tree, pos, labels=token_labels, font_size=10)

    plt.axis('off')
    plt.tight_layout()

    fig.savefig(os.path.join(args.model_dir, f"embeddings_{args.seed}.png"))

    # Extract the subgraph containing "User" tokens
    user_nodes = [node for node, label in token_labels.items() if 'User' in str(label)]
    user_subgraph = nx.subgraph(tree, user_nodes)
    # Visualize just this subgraph
