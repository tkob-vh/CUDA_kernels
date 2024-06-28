#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -w hepnode2
#SBATCH --output=output/conway.%j.out


echo "Starting job $SLURM_JOB_ID"
echo ${SLURM_JOB_NODELIST}
cat job.slurm

M=${M:="256"}
N=${N:="2048"}
case=${case:="answer"}
echo "M: $M"
echo "iterations: $N"
echo "case: $case"

. /data/spack/share/spack/setup-env.sh
spack load intel-oneapi-compilers@2024
spack load intel-oneapi-mkl
spack load cmake@3.27.9
spack load cuda@12.2.1


./$case data/input_$M.dat data/${case}_${M}.dat $N


# nsys profile --trace=cuda,nvtx \
#              --output=profiling_results/%q{SLURM_JOB_ID}.nsys-rep \
#              ./$case data/input_$M.dat data/${case}_${M}.dat $N


# nsys stats --report cuda_gpu_trace,cuda_gpu_kern_sum,cuda_api_sum \
#            ./profiling_results/${SLURM_JOB_ID}.nsys-rep

# ncu --kernel-name conway_step --launch-skip 90 --launch-count 1 "./${case}" data/input_$M.dat data/${case}_${M}.dat $N

# ncu --config-file off \
#     --export profiling_results/report%i \
#     --force-overwrite \
#     --kernel-name conway_step \
#     --launch-skip 89 \
#     --launch-count 1 \
#     --section-folder  profiling_results/sections \
#     ./$case data/input_$M.dat data/${case}_${M}.dat $N

echo "errors: $(cmp data/official_$M.dat data/${case}_${M}.dat -l | wc -l)"
# ===============================output============================
