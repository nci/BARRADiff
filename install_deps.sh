set -eu

module purge
module load python3/3.10.4

source env_vars

python3 -m venv $venv_dir
source $venv_dir/bin/activate

pip3 install -r requirements.txt

CURRENT_DIR=$PWD python3 - << 'EOF'
import os
curr_dir = os.environ.get('CURRENT_DIR', '.')
os.environ['HF_HOME'] = os.path.join(curr_dir, 'hf_cache')

from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
EOF

