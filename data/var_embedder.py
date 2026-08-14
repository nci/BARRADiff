#https://github.com/QwenLM/Qwen3-Embedding

import os
curr_dir = os.path.dirname(os.path.realpath(__file__))

if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.join(curr_dir, '../hf_cache')

import textwrap
import json
import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

class VarEmbed:
    def __init__(self, normalize=True, device=None):
        with open(os.path.join(curr_dir, 'barra2_vars_primary.json')) as f:
            self.barra2_data = json.load(f)

        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval().to(device)
        
        self.normalize = normalize

    @torch.inference_mode()
    def embed(self, var_name, var_level):
        var_rec = None
        for rec in self.barra2_data:
            if 'Available For (model/domain_id/freq)' not in rec:
                continue

            if 'AUST-04/1hr' not in rec['Available For (model/domain_id/freq)']:
                continue

            if var_name in rec['Variable']:
                var_rec = rec
                break
        
        assert var_rec is not None, f'{var_name} not found'

        var_doc = ' '.join([var_name, str(var_level), rec['Long name (Units)'], rec['Description']])
        embedding = self.model.encode([var_doc,], convert_to_tensor=True)
        
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
    
if __name__ == '__main__':
    embed = VarEmbed()
    
    import time
    t0 = time.time()
    embedding = embed.embed('ps', None)
    t1 = time.time() - t0
    print(embedding)
    print(embedding.shape, f'time={t1:.3f}')
    
    embedding2 = embed.embed('ta200', 200)
    print(embed.model.similarity(embedding, embedding2))
    
