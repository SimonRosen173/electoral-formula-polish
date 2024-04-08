#!/bin/bash
# specify a partition
#SBATCH -p stampede
# specify number of nodes
#SBATCH -N 1
# specify the job name
#SBATCH -J exp2a
# specify the filename to be used for writing output
# NOTE: You must replace the username `hwang' with your own account name!!
#SBATCH -o /datasets/srosen/logs/elec/slurms/slurm_exp2a.%N.%j.out
# specify the filename for stderr
#SBATCH -e /datasets/srosen/logs/elec/slurms/slurm_exp2a.%N.%j.err

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

python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 0 > \datasets\srosen\logs\elec\runs\exp2a/run_0.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 1 > \datasets\srosen\logs\elec\runs\exp2a/run_1.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 2 > \datasets\srosen\logs\elec\runs\exp2a/run_2.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 3 > \datasets\srosen\logs\elec\runs\exp2a/run_3.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 4 > \datasets\srosen\logs\elec\runs\exp2a/run_4.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 5 > \datasets\srosen\logs\elec\runs\exp2a/run_5.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 6 > \datasets\srosen\logs\elec\runs\exp2a/run_6.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 7 > \datasets\srosen\logs\elec\runs\exp2a/run_7.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 8 > \datasets\srosen\logs\elec\runs\exp2a/run_8.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 9 > \datasets\srosen\logs\elec\runs\exp2a/run_9.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 10 > \datasets\srosen\logs\elec\runs\exp2a/run_10.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 11 > \datasets\srosen\logs\elec\runs\exp2a/run_11.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 12 > \datasets\srosen\logs\elec\runs\exp2a/run_12.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 13 > \datasets\srosen\logs\elec\runs\exp2a/run_13.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 14 > \datasets\srosen\logs\elec\runs\exp2a/run_14.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 15 > \datasets\srosen\logs\elec\runs\exp2a/run_15.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 16 > \datasets\srosen\logs\elec\runs\exp2a/run_16.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 17 > \datasets\srosen\logs\elec\runs\exp2a/run_17.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 18 > \datasets\srosen\logs\elec\runs\exp2a/run_18.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 19 > \datasets\srosen\logs\elec\runs\exp2a/run_19.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 20 > \datasets\srosen\logs\elec\runs\exp2a/run_20.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 21 > \datasets\srosen\logs\elec\runs\exp2a/run_21.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 22 > \datasets\srosen\logs\elec\runs\exp2a/run_22.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 23 > \datasets\srosen\logs\elec\runs\exp2a/run_23.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 24 > \datasets\srosen\logs\elec\runs\exp2a/run_24.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 25 > \datasets\srosen\logs\elec\runs\exp2a/run_25.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 26 > \datasets\srosen\logs\elec\runs\exp2a/run_26.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 27 > \datasets\srosen\logs\elec\runs\exp2a/run_27.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 28 > \datasets\srosen\logs\elec\runs\exp2a/run_28.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 29 > \datasets\srosen\logs\elec\runs\exp2a/run_29.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 30 > \datasets\srosen\logs\elec\runs\exp2a/run_30.out 2>&1 &
python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --run C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\configs\exp2a.json --folder /tmp/myrun/data -bn 31 > \datasets\srosen\logs\elec\runs\exp2a/run_31.out 2>&1 &


wait

echo All processes finished

python3 C:\Users\simon\Documents\Work\IEC\Code\electoral-formula-polish\electoral_formula\api.py --join /tmp/myrun/data --n-batches 32 --formulae amend_orig
zip -r /tmp/myrun/data.zip /tmp/myrun/data/agr/*
mv /tmp/myrun/data.zip \datasets\srosen\logs\elec\data\exp2a/data.zip

echo Cleaned up and done :P
