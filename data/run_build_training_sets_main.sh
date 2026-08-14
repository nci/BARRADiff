set -eu

source ../env_vars

module purge
module load python3/3.10.4
export PATH=$venv_dir/bin:$PATH

python gen_pbs_scripts.py

script_files=$(ls build_training_set_scripts/run_build_sharded_training_set_*.sh)

for file in $script_files
do
    echo $file
    
    (
        set -eu
        cd ../
        qsub data/$file
    )
done
