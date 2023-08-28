#!/bin/bash

#SBATCH --job-name=Periodic			#name of the job to find when calling >>>sacct or >>>squeue
#SBATCH --nodes=1				#number of nodes, i.e. computers, to request off the cluster (nodes typically have ~20 singled threaded cores)
#SBATCH --ntasks=1				#how many independent script you are hoping to run 
#SBATCH --cpus-per-task=10			#how many threads to multithread across (no point more than number of cores available. also, you cannot thread across nodes) 
#SBATCH --time=24:00:00				#compute time
#SBATCH --mem=16gb				#memory to request
#SBATCH --output=./logs/%J.log		#where to save output log files (julia script prints here
#SBATCH --error=./logs/%J.err		#where to save output error files         #task ID array for array scripting (can be passed to script below with command line argument --slurm-array-task-id $SLURM_ARRAY_TASK_ID)

pwd; hostname;

python -u train.py

sstat -j $SLURM_JOB_ID.batch --format=JobID,MaxVMSize,
