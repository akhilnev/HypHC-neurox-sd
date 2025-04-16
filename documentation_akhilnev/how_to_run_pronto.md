# How to Run HypHC on Nova HPC Environment at Iowa State University

This guide provides step-by-step instructions for running the Hyperbolic Hierarchical Clustering (HypHC) code on the Nova HPC environment at Iowa State University.

## 1. Accessing Nova

1. **First-time Setup**:
   * Install an authenticator app on your mobile device (Google Authenticator or MS Authenticator)
   * Follow the instructions at [ISU HPC Cluster Access](https://research.it.iastate.edu/accessing-clusters#with-ssh)

2. **Login to Nova**:
   ```bash
   ssh <your-netid>@nova.its.iastate.edu
   ```
   * Enter your ISU password when prompted
   * Enter the 6-digit verification code from your authenticator app when prompted

3. **Off-campus Access**:
   * Install and connect to the [ISU VPN](https://www.it.iastate.edu/services/vpn) when connecting from off-campus

4. **Create a Working Directory**:
   ```bash
   mkdir -p /work/<group>/<your-netid>/HypHC
   cd /work/<group>/<your-netid>/HypHC
   ```
   * Replace `<group>` with your research group name (run `groups` command to see your group)

## 2. Setting Up the Environment

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HazyResearch/HypHC.git .
   ```

2. **Load the HypHC Module**:
   ```bash
   # Access the modules directory
   module use /ptmp/opt/rit/x86_64/gcc-11.4/modules
   
   # Load the HypHC module
   module load hyphc/benchmarks
   
   # Display module instructions
   note_4_demo
   ```

3. **Create a Micromamba Environment**:
   ```bash
   # Follow the output from note_4_demo to create the environment
   micromamba create -p $PWD/micromamba/envs/hyphc_env -c conda-forge pytorch=1.6.0 python=3.7 cudatoolkit=10.2
   eval "$(micromamba shell hook --shell=bash)"
   micromamba activate $PWD/micromamba/envs/hyphc_env
   ```

4. **Install Additional Dependencies**:
   ```bash
   pip install networkx==2.2 numpy==1.19.2 tqdm==4.31.1 scikit-learn cython==0.29.21
   
   # Build required C extensions
   cd mst; python setup.py build_ext --inplace; cd ..
   cd unionfind; python setup.py build_ext --inplace; cd ..
   ```

5. **Set Environment Variables**:
   ```bash
   export HHC_HOME=$(pwd)
   export DATAPATH="$HHC_HOME/data"
   export SAVEPATH="$HHC_HOME/embeddings"
   ```

## 3. Preparing Your Data

1. **Create Data Directory**:
   ```bash
   mkdir -p data/processed_activations/
   ```

2. **Transfer Your Data**:
   ```bash
   # Option 1: Using scp from your local machine
   scp /path/to/local/_processed_*.npy <your-netid>@novadtn.its.iastate.edu:/work/<group>/<your-netid>/HypHC/data/processed_activations/

   # Option 2: Copy from another location on Nova
   cp /path/to/embeddings/_processed_*.npy data/processed_activations/
   ```
   
   Note: Use the data transfer node (`novadtn.its.iastate.edu`) for file transfers.

## 4. Running Jobs on Nova

### Interactive Session

1. **Request an Interactive Session with GPU**:
   ```bash
   # Request a GPU node (V100 or A100)
   salloc -N1 -n8 -t4:0:0 --mem=64G --gres=gpu:1
   ```

2. **On the Compute Node**:
   ```bash
   # Load the module
   module use /ptmp/opt/rit/x86_64/gcc-11.4/modules
   module load hyphc/benchmarks
   
   # Activate your environment
   eval "$(micromamba shell hook --shell=bash)"
   micromamba activate $PWD/micromamba/envs/hyphc_env
   
   # Set environment variables
   export HHC_HOME=$(pwd)
   export DATAPATH="$HHC_HOME/data"
   export SAVEPATH="$HHC_HOME/embeddings"
   
   # Run the training script
   ./examples/run_codebert.sh
   ```

### Batch Job Submission

1. **Create a Job Script** (save as `run_hyphc.sh`):
   ```bash
   #!/bin/bash
   #SBATCH -N 1                     # Request 1 node
   #SBATCH -n 8                     # Request 8 cores
   #SBATCH -t 8:00:00               # Request 8 hours
   #SBATCH --mem=64G                # Request 64GB memory
   #SBATCH --gres=gpu:1             # Request 1 GPU
   #SBATCH -J HypHC                 # Job name
   #SBATCH -o hyphc_%j.out          # Output file
   #SBATCH -e hyphc_%j.err          # Error file
   
   # Change to your working directory
   cd /work/<group>/<your-netid>/HypHC
   
   # Load the module
   module use /ptmp/opt/rit/x86_64/gcc-11.4/modules
   module load hyphc/benchmarks
   
   # Activate your environment
   eval "$(micromamba shell hook --shell=bash)"
   micromamba activate $PWD/micromamba/envs/hyphc_env
   
   # Set environment variables
   export HHC_HOME=$(pwd)
   export DATAPATH="$HHC_HOME/data"
   export SAVEPATH="$HHC_HOME/embeddings"
   
   # Run the training script with modified parameters for larger datasets
   python train.py --dataset codebert \
                  --epochs 50 \
                  --batch_size 128 \
                  --learning_rate 1e-3 \
                  --temperature 1e-1 \
                  --eval_every 1 \
                  --patience 30 \
                  --optimizer RAdam \
                  --anneal_every 50 \
                  --anneal_factor 0.5 \
                  --init_size 5e-2 \
                  --num_samples 100000 \
                  --seed 0
   ```

2. **Submit the Job**:
   ```bash
   sbatch run_hyphc.sh
   ```

3. **Monitor Your Job**:
   ```bash
   squeue -u <your-netid>
   ```

## 5. Analyzing Results

1. **Extract Clusters**:
   ```bash
   python extract_subclusters.py --model_dir embeddings/codebert/<hash> --seed 0 --min_size 2
   ```

2. **Visualize Tree** (with X11 forwarding):
   ```bash
   python visualize_tree.py --model_dir embeddings/codebert/<hash> --seed 0
   ```

3. **Copy Results Back to Local Machine**:
   ```bash
   # From your local machine
   scp -r <your-netid>@nova.its.iastate.edu:/work/<group>/<your-netid>/HypHC/embeddings/codebert/<hash> /local/path/
   ```

## 6. Memory Optimization for Large Datasets

If you encounter memory issues when running with large datasets, consider these adjustments:

1. **Reduce Batch Size**:
   Modify `run_hyphc.sh` to use a smaller batch size (e.g., 64 or 32 instead of 128/256).

2. **Reduce Number of Samples**:
   Decrease the `--num_samples` parameter (e.g., 50000 instead of 100000).

3. **Request More Memory**:
   Increase the `--mem` parameter in your SBATCH script (e.g., `--mem=128G`).

4. **Use Data Subsampling**:
   If your dataset is very large, consider using a subset for initial experiments.

## 7. Benchmarks

Based on the information from the IT staff (Xuefeng Zhao), here are the expected runtimes:

- **A100 GPU**: ~11 minutes total
  - ~4 minutes for training
  - ~7 minutes for A100 to convert/load legacy code
  
- **V100 GPU**: ~6 minutes total

Note: For the first ~7 minutes on A100, it may appear that the code is hanging, but it's actually converting the legacy code for the A100 architecture.

## 8. Troubleshooting

1. **Module Not Found**:
   ```bash
   # Make sure to use the correct module path
   module use /ptmp/opt/rit/x86_64/gcc-11.4/modules
   module load hyphc/benchmarks
   ```

2. **CUDA Issues**:
   Ensure you're using the correct CUDA version (10.2) with PyTorch 1.6.0.

3. **Permission Issues**:
   Check that you have write permissions in your working directory and the embeddings folder.

4. **For Additional Help**:
   Contact ResearchIT at researchit@iastate.edu or your advisor.
