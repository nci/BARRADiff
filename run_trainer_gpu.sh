#!/bin/bash
#PBS -P fp0
#PBS -q gpuvolta
#PBS -l wd
#PBS -l walltime=48:00:00
#PBS -l mem=720GB
#PBS -l ncpus=96
#PBS -l ngpus=8
#PBS -l jobfs=200GB
#PBS -l storage=gdata/rt52+gdata/fr5+gdata/ob53+scratch/z00+scratch/fp0
#PBS -N pbs_edm2_train_multi_gpu

set -eu

source env_vars

module purge
module load python3/3.10.4
module load cuda/11.7.0
module load openmpi/4.1.4
module load nccl/2.13.4

export PATH=$venv_dir/bin:$PATH

batch_size=8
accum_steps=1
network_arch=edm2-img512-xxl
model_shards=4
experiment=00000-$network_arch
output_path=$training_run_dir/$experiment

if [ -z ${PBS_O_WORKDIR:-''} ]
then
  curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
else
  curr_dir="$PBS_O_WORKDIR"
fi

data_path=$training_data_dir
if [ ! -d $data_path ]
then
  echo "$data_path is not accessible"
  exit 1
fi

hostfile=${PBS_NODEFILE:-''}
if [ -z "$hostfile" ]
then
  hostfile=$(mktemp)
  echo $(hostname) > $hostfile
fi

num_nodes=${PBS_NNODES:-$(cat $hostfile|sort -u|wc -l)}
if [ $num_nodes -eq 1 ]
then
  local_world_size=${PBS_NGPUS:-1}
else
  local_world_size=4
fi

function cleanup {
  set +e
  rm -f $hostfile
  set -e
}
trap "cleanup" EXIT

next_job_file=$(mktemp)
(
cd $curr_dir
PBS_JOBID=${PBS_JOBID:-''}
if [[ ! -z $PBS_JOBID ]]
then
  set +e
  # Uncomment this for self-submitting training jobs
  #qsub -W depend=afterany:$PBS_JOBID run_trainer_gpu.sh > $next_job_file &
  set -e
fi

)

(
cd $curr_dir

mpirun -np $num_nodes \
  --hostfile $hostfile \
  -bind-to none --map-by node \
  -x LD_LIBRARY_PATH -x PATH \
  -x NCCL_SOCKET_NTHREADS=4 -x NCCL_NSOCKS_PERTHREAD=4 \
  -mca pml ob1 -mca btl ^openib \
  torchrun \
    --nnodes=$num_nodes \
    --nproc_per_node=$local_world_size \
    --rdzv_id=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$(head -n1 $hostfile):29601" \
    train_edm2.py \
    --outdir=$output_path \
    --data=$data_path \
    --preset=$network_arch \
    --batch-gpu=$batch_size \
    --grad-accumulation=$accum_steps \
    --model-shards=$model_shards

)

