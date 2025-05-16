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

## 9. Disconnecting and Reconnecting

### Disconnecting from Nova

When you need to leave your work and disconnect from Nova, you can do so safely with these steps:

1. **If in an Interactive Session**:
   ```bash
   # Exit from the interactive compute node
   exit
   ```

2. **Exit from Nova Login Node**:
   ```bash
   # Exit from Nova
   exit
   ```

Your batch jobs will continue to run in the background even after you log out.

### Reconnecting to Nova and Resuming Work

When you're ready to continue your work, follow these steps to get back to where you left off:

1. **Login to Nova**:
   ```bash
   ssh <your-netid>@nova.its.iastate.edu
   ```
   * Enter your password and verification code when prompted

2. **Navigate to Your Working Directory**:
   ```bash
   cd /work/classtmp/akhilnev/HypHC
   ```

3. **Check Job Status**:
   ```bash
   # Check the status of any running jobs
   squeue -u akhilnev
   
   # Check details of completed jobs
   sacct -u akhilnev --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
   ```

4. **View Job Output**:
   ```bash
   # View output from your job (replace with your job ID)
   cat logs/slurm-JOBID.out
   ```

5. **Reactivate Your Environment**:
   ```bash
   # Initialize conda
   source /work/classtmp/akhilnev/miniconda3/bin/activate
   
   # Activate your HypHC environment
   conda activate hyphc_env
   
   # Set environment variables
   export HHC_HOME=$(pwd)
   export DATAPATH="$HHC_HOME/data"
   export SAVEPATH="$HHC_HOME/embeddings"
   ```

6. **Continue Your Work**:
   * Submit new jobs if needed
   * Analyze results from completed jobs
   * Make adjustments based on previous job outcomes

### Checking for Results

If your job has completed while you were away:

1. **Check for Generated Model Files**:
   ```bash
   ls -la embeddings/codebert/
   ```

2. **Analyze the Results**:
   ```bash
   # Find the most recent output directory
   MODEL_DIR=$(ls -t embeddings/codebert/ | head -1)
   
   # Extract clusters
   python extract_subclusters.py --model_dir embeddings/codebert/$MODEL_DIR --seed 0 --min_size 2
   ```

3. **Transfer Results to Local Machine** (if needed):
   ```bash
   # From your local machine
   scp -r akhilnev@novadtn.its.iastate.edu:/work/classtmp/akhilnev/HypHC/embeddings/codebert/$MODEL_DIR /path/on/local/machine/

## 10. Challenges Encountered

After successfully running the model, we encountered several challenges during the analysis phase:

### Memory Issues with Visualization

When attempting to analyze the HypHC results locally (on a MacBook), we experienced severe memory limitations:

- Running `visualize.py` with a dataset of 47,321 tokens crashed due to memory exhaustion
- The script began calculating similarities between points but was killed by the operating system
- Similarity calculation appeared to be processing in batches (1000 tokens at a time) but still exhausted memory

We attempted to create a memory-optimized script (`extract_clusters_to_file.py`) that:
- Used garbage collection
- Processed branches individually
- Limited token display
- Wrote directly to file

However, even this optimized approach crashed during the tree decoding phase before it could complete.

### Long Processing Times on Nova

When running analysis scripts on Nova:

- The `extract_clusters_to_file.py` script began executing but took over 30 minutes without completing
- The most time-consuming step was "Decoding tree" which transforms embeddings into a hierarchical structure
- The process appeared to stall with no progress indicators during tree decoding

Our SSH connection to Nova disconnected during the long-running process, terminating the extraction job before completion.

### SSH Connection Stability

Issues with maintaining connection to Nova:

- Interactive sessions were terminated when SSH connections dropped
- Long-running processes (30+ minutes) were vulnerable to connection interruptions
- Laggy terminal response made monitoring progress difficult

### Alternative Analysis Approach

We created a lightweight analysis script that:
- Sampled 3,000 tokens instead of using all 47,321
- Analyzed token neighborhoods based on embedding similarity
- Avoided the memory-intensive tree decoding step
- Generated statistics about the embedding space

This approach completed successfully but provided limited insight without the full hierarchical clustering structure.

### Unexpected Embedding Properties

The embedding analysis revealed:
- All tokens appeared to have identical norms (0.6812)
- This suggests tokens are positioned equidistant from the origin in hyperbolic space
- This is unusual for hyperbolic embeddings, which typically show varying distances from the origin

### Batch Job Limitations

Attempts to use SLURM batch jobs faced challenges:
- Initial quality of service (QoS) configuration was rejected
- Required identifying available QoS options (instruction and scavenger) through `sacctmgr`
- Needed to adapt script parameters to match available resources

### X11 Requirements

When attempting visualization:
- Scripts required X11 forwarding capabilities
- Headless plotting mode was needed but not initially configured
- Additional dependencies like matplotlib and pillow were required
   ```
