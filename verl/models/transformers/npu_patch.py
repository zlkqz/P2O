

import torch
import torch_npu
from torch_npu import npu_rotary_mul as apply_rotary_emb
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm


def apply_rotary_pos_emb_flashatt_npu(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    cos = cos.repeat(1, 2)
    sin = sin.repeat(1, 2)
    q_embed = apply_rotary_emb(
        q.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()
    ).type_as(q)
    k_embed = apply_rotary_emb(
        k.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()
    ).type_as(k)
    return q_embed, k_embed


def rms_norm_forward(self, x):
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


Qwen2RMSNorm.forward = rms_norm_forward
modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = apply_rotary_pos_emb_flashatt_npu
