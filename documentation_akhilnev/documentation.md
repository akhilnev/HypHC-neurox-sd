# Integrating CodeBERT Activations with HypHC

## Overview
This adaptation enables HypHC to work with pre-computed embeddings from CodeBERT, 
allowing hierarchical clustering of code tokens based on their semantic similarities.

## Data Preparation
- _processed_points.npy: Contains CodeBERT activation vectors
- _processed_vocab.npy: Contains corresponding vocabulary tokens

## Implementation Changes

1. Added custom data loader in datasets/loading.py:
   - New function load_codebert_data() to load .npy files
   - Modified load_data() to handle "codebert" dataset type
   - Implemented cosine similarity calculation for embeddings

2. Created custom run script (run_codebert.sh):
   - Added appropriate hyperparameters for code embeddings
   - Adjusted batch size and sampling for larger dataset

3. Modified visualization in visualize.py:
   - Added special case for CodeBERT vocabulary
   - Implemented token sampling for clearer visualization
   - Added filtering options for meaningful subsets

4. Memory optimizations:
   - Added sparse similarity matrix support
   - Implemented batched processing for large datasets

## Usage
1. Place activation files in data/processed_activations/
2. Run with: ./examples/run_codebert.sh
3. View results with: python visualize.py --model_dir [output_dir] --seed 0

## Environment Setup

Before running the scripts, set the SAVEPATH environment variable to specify where 
output files should be saved:

```bash
export SAVEPATH=/Users/akhileshnevatia/HypHC/embeddings
```

Alternatively, modify utils/training.py to use a default path when SAVEPATH is not set.

## Dependencies

In addition to the original HypHC dependencies, this adaptation requires:

- scikit-learn: Used for computing cosine similarity between embeddings
  ```bash
  pip install scikit-learn
   ```

## Data Type Considerations

The HypHC implementation expects double-precision (64-bit) floats for similarity matrices.
When using CodeBERT activations, ensure that:

1. Similarity matrices are converted to np.float64 before being passed to dasgupta_cost
2. The data loader explicitly converts the similarity matrix using .astype(np.float64)

This prevents "Buffer dtype mismatch" errors during evaluation.


New scripts added to view sub-clusters ( cant view subgraphs as everything is one big connected component)

- extract_subclusters.py

- visualize_tree.py : to view the actual tree of tokens ( normal visualize shows us the same but in 3-D Space....)



