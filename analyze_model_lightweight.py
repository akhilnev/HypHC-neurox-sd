import argparse
import json
import os
import numpy as np
import torch
import gc  # Garbage collection
import random

def analyze_model_lightweight(model_dir, seed=0, output_file=None, sample_size=5000):
    """Extract basic information from model without full tree decoding"""
    print("Loading configuration and vocabulary...")
    
    # Set output file
    if output_file is None:
        output_file = os.path.join(model_dir, f"analysis_seed{seed}_sample{sample_size}.txt")
    
    # Load configuration
    config = json.load(open(os.path.join(model_dir, "config.json")))
    
    # Load vocabulary
    vocab = np.load("data/processed_activations/_processed_vocab.npy", allow_pickle=True)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(model_dir, f"model_{seed}.pkl")
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    
    with open(output_file, 'w') as f:
        f.write(f"Analysis of model {model_dir}, seed {seed}\n\n")
        
        # Extract and write model configuration
        f.write("== Model Configuration ==\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Sample a subset of tokens to analyze
        if len(vocab) > sample_size:
            indices = random.sample(range(len(vocab)), sample_size)
        else:
            indices = range(len(vocab))
        
        # Get embeddings
        print("Loading model class...")
        from model.hyphc import HypHC
        model = HypHC(len(vocab), config.get("rank", 2), config.get("temperature", 0.1), 
                     config.get("init_size", 0.05), config.get("max_scale", 1.0))
        model.load_state_dict(model_state)
        
        # Get embeddings
        print("Extracting embeddings...")
        embeddings = model.normalize_embeddings(model.embeddings.weight.data).detach().cpu().numpy()
        
        # Free memory
        del model_state
        del model
        gc.collect()
        
        # Find similar token groups without full tree construction
        print("Finding similar tokens based on embedding distances...")
        f.write("== Sample Token Neighborhoods ==\n")
        
        # Analyze sample of tokens and find their neighbors
        for i, idx in enumerate(indices[:100]):  # Only look at 100 samples
            if i % 10 == 0:
                print(f"Processing token {i}/100...")
                
            token = vocab[idx]
            embedding = embeddings[idx]
            
            # Calculate distance to other tokens in our sample
            dists = []
            for j, other_idx in enumerate(indices):
                if other_idx != idx:
                    # Use simple Euclidean distance for efficiency
                    dist = np.linalg.norm(embedding - embeddings[other_idx])
                    dists.append((dist, other_idx))
            
            # Sort by distance
            dists.sort()
            
            # Get 5 closest neighbors
            f.write(f"\nToken: {token}\n")
            f.write("Closest neighbors:\n")
            for j in range(min(5, len(dists))):
                neighbor_idx = dists[j][1]
                f.write(f"  {vocab[neighbor_idx]} (distance: {dists[j][0]:.4f})\n")
            
            # Garbage collect periodically
            if i % 20 == 0:
                gc.collect()
        
        # Add information about embedding space
        f.write("\n== Embedding Space Statistics ==\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n")
        f.write(f"Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}\n")
        
        # Find most central tokens (closest to origin in hyperbolic space)
        norms = np.linalg.norm(embeddings, axis=1)
        central_indices = np.argsort(norms)[:20]
        
        f.write("\n== Most Central Tokens (closest to origin) ==\n")
        for idx in central_indices:
            f.write(f"  {vocab[idx]} (norm: {norms[idx]:.4f})\n")
        
        # Find most peripheral tokens (farthest from origin)
        peripheral_indices = np.argsort(norms)[-20:]
        
        f.write("\n== Most Peripheral Tokens (farthest from origin) ==\n")
        for idx in reversed(peripheral_indices):
            f.write(f"  {vocab[idx]} (norm: {norms[idx]:.4f})\n")
    
    print(f"Analysis complete! Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--seed", type=int, default=0, help="Model seed to use")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path")
    parser.add_argument("--sample_size", type=int, default=5000, help="Number of tokens to sample for analysis")
    
    args = parser.parse_args()
    
    analyze_model_lightweight(
        args.model_dir, 
        args.seed, 
        args.output_file, 
        args.sample_size
    )
