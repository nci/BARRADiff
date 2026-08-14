set -eu

source env_vars

module load python3/3.10.4
export PATH=$venv_dir/bin:$PATH

training_data_root=$training_data_dir/data

output_root=$training_data_dir/training_data_plots

files=$(ls $training_data_root)
for f in $files
do
    echo "$training_data_root/$f"
    out_f=$(python -c "print('.'.join('$f'.split('.')[:-2])) ")

    rm -rf $output_root/$out_f
    python plot_training_sample.py --data-root $training_data_root/$f --output-root $output_root/$out_f
done

