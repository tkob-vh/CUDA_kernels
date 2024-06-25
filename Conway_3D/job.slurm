#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -w hepnode2
#SBATCH --output=output/conway.%j.out


echo "Starting job $SLURM_JOB_ID"
echo ${SLURM_JOB_NODELIST}
cat job.slurm

N=${N:="1024"}
case=${case:="answer"}
echo "iterations: $N"
echo "case: $case"

. /data/spack/share/spack/setup-env.sh
spack load intel-oneapi-compilers@2024
spack load intel-oneapi-mkl
spack load cmake@3.27.9
spack load cuda@12.2.1


./$case data/input.dat data/$case.dat $N


# nsys profile --trace=cuda,nvtx \
#              --output=profiling_results/%q{SLURM_JOB_ID}.nsys-rep \
#              ./$case data/input.dat data/$case.dat $N


# nsys stats --report cuda_gpu_trace,cuda_gpu_kern_sum,cuda_api_sum \
#            --output ./profiling_results \
#            ./profiling_results/${SLURM_JOB_ID}.nsys-rep


echo "errors: $(cmp data/official.dat data/$case.dat -l | wc -l)"
# ===============================output============================
