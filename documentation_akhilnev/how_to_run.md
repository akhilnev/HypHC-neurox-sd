# How to Run HypHC with CodeBERT Embeddings

## Setup Steps

1. **Environment Setup**
```bash
# Clone the repository (if not already done)
git clone [repository_url]
cd HypHC

# Create and activate virtual environment
python3 -m venv hyphc_env
source hyphc_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install scikit-learn  # Required for cosine similarity

# Install additional packages
cd mst; python setup.py build_ext --inplace
cd ../unionfind; python setup.py build_ext --inplace
cd ..
```

2. **Set Environment Variables**
```bash
# Set environment variables by running
source set_env.sh

# Verify paths are set correctly
echo $HHC_HOME  # Should show your HypHC directory
echo $DATAPATH  # Should show $HHC_HOME/data
echo $SAVEPATH  # Should show $HHC_HOME/embeddings
```

3. **Prepare Your Data**
- Place your embedding files in the correct location:
```bash
mkdir -p data/processed_activations/
cp /path/to/your/embeddings/_processed_points.npy data/processed_activations/
cp /path/to/your/embeddings/_processed_vocab.npy data/processed_activations/
```

4. **Run Training**
```bash
# Run the training script
./examples/run_codebert.sh
```
- This will create a directory in `embeddings/codebert/[hash]` with:
  - `model_0.pkl`: The trained model
  - `config.json`: Configuration parameters
  - `train_0.log`: Training logs

5. **Visualize Results**
```bash
# For tree visualization (recommended for code tokens)
python visualize_tree.py --model_dir embeddings/codebert/[hash] --seed 0

# For 3D visualization
python visualize.py --model_dir embeddings/codebert/[hash] --seed 0
```

6. **Extract Clusters**
```bash
# To view subclusters of tokens
python extract_subclusters.py --model_dir embeddings/codebert/[hash] --seed 0 --min_size 2
```

## Important Files

1. `data/processed_activations/`:
   - `_processed_points.npy`: Your embedding vectors
   - `_processed_vocab.npy`: Corresponding vocabulary tokens

2. `examples/run_codebert.sh`: Training script with parameters:
   - batch_size: 256
   - epochs: 50
   - learning_rate: 0.001
   - temperature: 0.1
   - dataset: "codebert"

3. `embeddings/`: Directory where trained models are saved

## Troubleshooting

1. If you get permission errors:
   - Check that SAVEPATH is set correctly in set_env.sh
   - Ensure you have write permissions in the embeddings directory

2. If you get memory errors:
   - Reduce batch_size in run_codebert.sh
   - Use fast_decoding=True (should be default)

3. For "dataset not found" errors:
   - Verify your .npy files are in data/processed_activations/
   - Check file permissions

## Notes

- Training time depends on:
  - Size of your embedding files
  - Number of epochs
  - Batch size
  - Available GPU/CPU resources

- For large datasets, consider:
  - Increasing batch_size if you have GPU memory
  - Reducing num_samples for faster training
  - Using early stopping (controlled by patience parameter)