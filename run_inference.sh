#!/bin/bash
#PBS -P fp0
#PBS -q gpuvolta
#PBS -l wd
#PBS -l walltime=10:00:00
#PBS -l mem=90GB
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l jobfs=200GB
#PBS -l storage=gdata/rt52+gdata/fr5+gdata/ob53+scratch/z00+scratch/fp0
#PBS -N pbs_edm2_inference_gpu

set -eu

source env_vars

module purge
module load python3/3.10.4
export PATH=$venv_dir/bin:$PATH

network_arch=edm2-img512-xxl
ema_length=0.05
experiment=00000-$network_arch

training_run_root=$model_weight_dir/$experiment

stats_root=$training_data_dir/stats

output_root=$inference_run_dir/$experiment

var_list_file=(mktemp)
cat > $var_list_file <<-EOF
tas uas vas ps psl prw pr prra clt rldscs rlds CAPE
ta200 ua200 va200 wap200 zg200 hus200
ta500 ua500 va500 wap500 zg500 hus500
ta700 ua700 va700 wap700 zg700 hus700
ta850 ua850 va850 wap850 zg850 hus850
EOF

barra2_var_names=$(cat $var_list_file)
rm -f $var_list_file

var_time='2022-01-01T01:00:00'

# we can override the var list here for quick debug
#barra2_var_names="tas"

# The FSDP trainer saves sharded Pytorch checkpoint file format but edm2 inference expects 
# "snapshot" file format. So we need a file format conversion here. We only need this once
#mkdir -p $training_run_root/snapshots
#python checkpoint2snapshots.py \
#    --input-root $training_run_root \
#    --output-root $training_run_root/snapshots

for var_name in $barra2_var_names
do
    python generate_images.py \
        --net=$training_run_root/snapshots \
        --ema-length $ema_length \
        --var-name $var_name \
	--var-time $var_time \
        --guidance=2.5 \
        --stats-root=$stats_root \
        --outdir=$output_root
done

