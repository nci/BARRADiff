#!/bin/bash
#PBS -P ak77
#PBS -q normal
#PBS -l wd
#PBS -l walltime=40:00:00
#PBS -l mem=160GB
#PBS -l ncpus=40
#PBS -l jobfs=200GB
#PBS -l storage=gdata/rt52+gdata/fr5+gdata/ob53+scratch/z00
#PBS -N pbs_barra2_latent_training_set_multi_part

set -eu

source env_vars

module purge
module load python3/3.10.4
export PATH=$venv_dir/bin:$PATH

var_name=tas
num_samples_per_shard=20

output_root=$training_data_dir
tmp_output_root=${output_root}_${var_name}

for d in data stats
do
    mkdir -p $output_root/$d
done

rm -rf $tmp_output_root

for seed in 0 100 200 300 400
do
    python build_training_set.py \
        --output-root $tmp_output_root \
        --total-samples $num_samples_per_shard \
        --seed $seed \
        --var $var_name

    rm -rf $tmp_output_root/data_${seed}
    mv $tmp_output_root/data $tmp_output_root/data_${seed}
    rm -rf $tmp_output_root/tasks

    (
        set -eu
        cd $tmp_output_root/data_${seed}/$var_name

        zarr_zip_file=$var_name.$seed.zarr.zip

        rm -f $zarr_zip_file
        zip -q -r -1 $zarr_zip_file *

        python -c "import zarr; ds=zarr.open('$zarr_zip_file'); assert ds['input'].shape[0] > 0;"

	rm -f $output_root/data/$zarr_zip_file
        mv $zarr_zip_file $output_root/data/$zarr_zip_file
        rm -rf $tmp_output_root/data_${seed}
    ) &

done

cp -f $tmp_output_root/stats/${var_name}_*.pt $output_root/stats/

wait

rm -rf $tmp_output_root

