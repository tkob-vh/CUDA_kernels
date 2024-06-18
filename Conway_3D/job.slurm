#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=output/conway.%j.out


echo "Starting job $SLURM_JOB_ID"
echo ${SLURM_JOB_NODELIST}
cat job.slurm

N=${N:="2048"}
case=${case:="answer"}
echo "iterations: $N"
echo "case: $case"

. /data/spack/share/spack/setup-env.sh
spack load intel-oneapi-compilers@2024
spack load intel-oneapi-mkl
spack load cmake@3.27.9
spack load cuda@12.2.1

# ./official data/input.dat data/official.dat 2048

./$case data/input.dat data/$case.dat $N

echo "errors: $(cmp data/baseline.dat data/$case.dat -l | wc -l)"
# ===============================output============================
