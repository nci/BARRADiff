import os
import shutil
import time

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor

from torch_utils.logger import get_logger

STATES_CHECKPOINT = "training_states.pt"
MODEL_CHECKPOINT = "model_state_dict.pt"
EMA_CHECKPOINT = "ema_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
PARAMS = "params"


def get_latest_checkpoint_folder(path):
    max_num = None
    if not os.path.exists(path):
        return max_num
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            try:
                num = int(name)
                if max_num is None or num > max_num:
                    max_num = num
            except ValueError:
                pass  # Skip non-numeric folder names
    return max_num


class Checkpointer:
    def __init__(self, folder: str, dcp_api: bool = False):
        self.folder = folder
        self.dcp_api = dcp_api
        self.last_training_time = get_latest_checkpoint_folder(
            f"{folder}/{'dcp_api' if dcp_api else 'dtensor_api'}"
        )
        self.logger = get_logger(__name__)

    def is_empty(self):
        return self.last_training_time is None

    def load_model(self, model: nn.Module):
        last_model_checkpoint = (
            f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
            f"/{self.last_training_time}/{MODEL_CHECKPOINT}"
        )
        if torch.distributed.get_rank() == 0:
            self.logger.info(f'loading model checkpoint: {last_model_checkpoint}')
        self._load_model(last_model_checkpoint, model)

    def load_ema(self, emas: list[nn.Module]):
        last_model_dir = (
            f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
            f"/{self.last_training_time}"
        )
        for name in os.listdir(last_model_dir):
            if name.startswith(EMA_CHECKPOINT):
                ema_idx = int(name.split('.')[-1])
                ema_checkpoint_path = os.path.join(last_model_dir, name)
                if torch.distributed.get_rank() == 0:
                    self.logger.info(f'loading EMA[{ema_idx}] checkpoint: {ema_checkpoint_path}')
                self._load_model(ema_checkpoint_path, emas[ema_idx])
        
    def load_optim(self, model: nn.Module, opt: torch.optim.Optimizer):
        last_optim_checkpoint = (
            f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
            f"/{self.last_training_time}/{OPTIM_CHECKPOINT}"
        )
        if torch.distributed.get_rank() == 0:
            self.logger.info(f'loading optimizer states: {last_optim_checkpoint}')
        full_sd = torch.load(
            last_optim_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )
        if self.dcp_api:
            set_optimizer_state_dict(
                model=model,
                optimizers=opt,
                optim_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return
        #_init_optim_state(opt)
        param_groups = opt.state_dict()["param_groups"]
        state = opt.state_dict()["state"]

        full_param_groups = full_sd["param_groups"]
        full_state = full_sd["state"]

        for param_group, full_param_group in zip(param_groups, full_param_groups):
            for key, value in full_param_group.items():
                if key == PARAMS:
                    continue
                param_group[key] = value
            for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
                if pid not in state:
                    continue
                param_state = state[pid]
                full_param_state = full_state[full_pid]
                for attr, full_tensor in full_param_state.items():
                    sharded_tensor = param_state[attr]
                    if isinstance(sharded_tensor, DTensor):
                        # exp_avg is DTensor
                        param_state[attr] = distribute_tensor(
                            full_tensor,
                            sharded_tensor.device_mesh,
                            sharded_tensor.placements,
                        )
                    else:
                        # step is plain tensor
                        param_state[attr] = full_tensor
        opt.load_state_dict(
            {
                "param_groups": param_groups,
                "state": state,
            }
        )

    def _load_model(self, model_checkpoint: str, model: nn.Module):
        full_sd = torch.load(
            model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )
        if self.dcp_api:
            set_model_state_dict(
                model=model,
                model_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return
        meta_sharded_sd = model.state_dict()
        sharded_sd = {}
        for param_name, full_tensor in full_sd.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if isinstance(sharded_meta_param, DTensor):
                sharded_tensor = distribute_tensor(
                    full_tensor,
                    sharded_meta_param.device_mesh,
                    sharded_meta_param.placements,
                )
            else:
                sharded_tensor = full_tensor.to(sharded_meta_param.device)
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        # choose `assign=True` since we cannot call `copy_` on meta tensor
        model.load_state_dict(sharded_sd, strict=False, assign=True)
        
    def _get_full_model_state_dict(self, model: nn.Module):
        if self.dcp_api:
            return get_model_state_dict(
                model=model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )

        sharded_sd = model.state_dict()
        cpu_state_dict = {}
        for param_name, sharded_param in sharded_sd.items():
            if isinstance(sharded_param, DTensor):
                full_param = sharded_param.full_tensor()
            else:
                full_param = sharded_param.clone()

            if torch.distributed.get_rank() == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param
        return cpu_state_dict

    def _get_full_optimizer_state_dict(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
    ):
        if self.dcp_api:
            return get_optimizer_state_dict(
                model=model,
                optimizers=opt,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        is_rank_zero = torch.distributed.get_rank() == 0
        sharded_sd = opt.state_dict()
        sharded_state = sharded_sd["state"]
        full_state = {}
        for group_id, sharded_group in sharded_state.items():
            group_state = {}
            for attr, sharded_tensor in sharded_group.items():
                if isinstance(sharded_tensor, DTensor):
                    # "exp_avg" in AdamW is `DTensor`
                    full_tensor = sharded_tensor.full_tensor()
                else:
                    # "step" in AdamW is plain tensor
                    full_tensor = sharded_tensor.clone()
                if is_rank_zero:
                    group_state[attr] = full_tensor.cpu()
                else:
                    del full_tensor
            if is_rank_zero:
                full_state[group_id] = group_state
            else:
                del group_state
        if is_rank_zero:
            return {
                "param_groups": sharded_sd["param_groups"],
                "state": full_state,
            }
        else:
            return {}

    def save(self, step: int, model: nn.Module, emas: list[nn.Module], optim: torch.optim.Optimizer, training_states):
        is_rank_zero = torch.distributed.get_rank() == 0
        new_checkpoint_folder = f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}/{step}"
        
        tmp_checkpoint_folder = new_checkpoint_folder + '_tmp'
        
        if is_rank_zero:
            os.makedirs(tmp_checkpoint_folder, exist_ok=True)
            states_checkpoint_path = f"{tmp_checkpoint_folder}/{STATES_CHECKPOINT}"
            self.logger.info(f'saving training states: {states_checkpoint_path}')
            torch.save(training_states, states_checkpoint_path)
        
        model_state_dict = self._get_full_model_state_dict(model)
        if is_rank_zero:
            model_checkpoint_path = f"{tmp_checkpoint_folder}/{MODEL_CHECKPOINT}"
            self.logger.info(f'saving model states: {model_checkpoint_path}')
            torch.save(model_state_dict, model_checkpoint_path)
        del model_state_dict
        
        for ii, ema in enumerate(emas):
            model_state_dict = self._get_full_model_state_dict(ema)
            if is_rank_zero:
                model_checkpoint_path = f"{tmp_checkpoint_folder}/{EMA_CHECKPOINT}.{ii}"
                self.logger.info(f'saving EMA[{ii}] states: {model_checkpoint_path}')
                torch.save(model_state_dict, model_checkpoint_path)
            
            del model_state_dict
        
        optim_state_dict = self._get_full_optimizer_state_dict(model, optim)
        if is_rank_zero:
            new_optim_checkpoint = f"{tmp_checkpoint_folder}/{OPTIM_CHECKPOINT}"
            self.logger.info(f'saving optimizer states: {new_optim_checkpoint}')
            torch.save(optim_state_dict, new_optim_checkpoint)
        del optim_state_dict
        
        if is_rank_zero:
            if os.path.exists(new_checkpoint_folder):
                shutil.rmtree(new_checkpoint_folder)
            shutil.move(tmp_checkpoint_folder, new_checkpoint_folder) 

    def load(self, model: nn.Module, emas: list[nn.Module], optim: torch.optim.Optimizer, training_states):
        self.load_model(model)
        self.load_ema(emas)
        self.load_optim(model, optim)
        
        last_states_checkpoint = (
            f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
            f"/{self.last_training_time}/{STATES_CHECKPOINT}"
        )
        last_states = torch.load(
            last_states_checkpoint, weights_only=False, map_location="cpu"
        )
        training_states.clear()
        training_states.update(last_states)
        
        
