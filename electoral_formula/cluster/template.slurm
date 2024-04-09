#!/bin/bash
# specify a partition
#SBATCH -p {partition}
# specify number of nodes
#SBATCH -N 1
# specify the job name
#SBATCH -J {job_name}
# specify the filename to be used for writing output
# NOTE: You must replace the username `hwang' with your own account name!!
#SBATCH -o {slurm_logs_folder}/slurm_{job_name}.%N.%j.out
# specify the filename for stderr
#SBATCH -e {slurm_logs_folder}/slurm_{job_name}.%N.%j.err

echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------
cd $SLURM_SUBMIT_DIR
pwd

rm -rf /tmp/myrun/data
mkdir /tmp/myrun/data -p

source activate elec

cd /home-mscluster/srosen/code/electoral-formula-polish/electoral_formula

{run_commands}

wait

echo All processes finished

{finish_commands}

rm -rf /tmp/myrun/data

echo Cleaned up and done :P
