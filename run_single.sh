#!/bin/bash
#SBATCH --output=logs/%j_%x.out      # Log file named with JobID_JobName
#SBATCH --error=logs/%j_%x.err       # Error file named with JobID_JobName
#SBATCH --time=11:59:00              # 12-hour limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # 8 cores for Gurobi
#SBATCH --mem=32G                    # 32 GB RAM

# Load modules
module load miniforge/25.11.0-0
module load gurobi/12.0.3

# Activate environment
source activate scnd_env

# Run the python script that was passed as an argument
python -u $1
