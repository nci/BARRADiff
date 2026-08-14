import os
import shutil
from training_data import get_c2_training_pair_templates

if __name__ == '__main__':
    with open(f'run_build_sharded_training_set.sh.tpl') as f:
        script_tpl = f.read()

    templates = get_c2_training_pair_templates()
    
    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_training_set_scripts')
    
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    for it, tpl in enumerate(templates):
        barra2_var_name = tpl[-1].split('/')[-1].split('_')[0]
        
        script = script_tpl.replace('{{var_name}}', barra2_var_name)
        script_file_path = os.path.join(output_root, f'run_build_sharded_training_set_{barra2_var_name}.sh')
        print(f'{it+1}/{len(templates)}    {script_file_path}')
        with open(script_file_path, 'w') as f:
            f.write(script)
