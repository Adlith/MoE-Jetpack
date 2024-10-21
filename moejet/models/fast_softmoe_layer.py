import math
from typing import Optional, Union

import torch
from einops import einsum, rearrange
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union
from timm.layers import PatchEmbed, Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat, \
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

class OffsetScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = x * self.gamma + self.beta
        return out
    
class ExpertLayerNorm(nn.Module):
    def __init__(self, norm_weight, norm_bias):
        super().__init__()
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        self.eps = 1e-6
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        norm_weight = rearrange(self.norm_weight, "e d2 -> () e () d2")
        norm_bias = rearrange(self.norm_bias, "e d2 -> () e () d2")
        return x * norm_weight + norm_bias
    
class BlockExpert_LayerNorm(nn.Module):
    def __init__(self, num_experts, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((num_experts, dim)))
        self.bias = nn.Parameter(torch.zeros((num_experts, dim)))
        self.eps = 1e-6
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        weight = rearrange(self.weight, "e d2 -> () e () d2")
        bias = rearrange(self.bias, "e d2 -> () e () d2")
        return x * weight + bias
    
class MultiExpertLinear(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, use_bias=True):
        super(MultiExpertLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_dim, input_dim))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)
        # 初始化权重和偏置
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        x = einsum(x, self.weight, "b e s d1, e d2 d1 -> b e s d2")
        x = x + rearrange(self.bias, "e d2 -> () e () d2") if self.use_bias else x
        return x

class MultiExpertConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, kernel_size, stride=1, padding=0, dilation=1, use_bias=True):
        super(MultiExpertConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (num_experts, kernel_size, kernel_size)
        self.stride = (1, stride, stride)
        self.padding = (0, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        
        # 可学习的权重参数
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # 使用手动定义的参数进行卷积操作
        x = F.conv3d(x, self.weight, None, self.stride, self.padding, self.dilation)
        x = x + rearrange(self.bias, "e d -> () d e () ()") if self.use_bias else x
        return x

class MultiExpertLayer(nn.Module):
    """A more efficient alternative to creating 'n' separate expert layers (likely
    from 'nn.Linear' modules).  Instead, we create a single set of batched weights
    and biases, and apply all 'experts' in parallel.

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        bias (bool): whether to include a bias term. Default: True
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_experts: int,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        act_fn: nn.Module = nn.GELU,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        norm_layer=None,
        layer_scale=False,
        glu=False,
        freeze_moe=False,
    ):
        super().__init__()
        self.in_features = in_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        
        self.glu = glu
        if self.glu:
            self.eva_fc1_g = MultiExpertLinear(num_experts, in_dim, self.hidden_dim)
            self.eva_fc1_x = MultiExpertLinear(num_experts, in_dim, self.hidden_dim)
            self.act_fn = nn.SiLU()
            self.eva_norm = BlockExpert_LayerNorm(num_experts, self.hidden_dim)
            self.eva_fc2 = MultiExpertLinear(num_experts, self.hidden_dim, in_dim)
        else:
            self.vit_fc1 = MultiExpertLinear(num_experts, in_dim, self.hidden_dim)
            self.act_fn = act_fn()
            self.vit_fc2 = MultiExpertLinear(num_experts, self.hidden_dim, in_dim)
        
        self.layer_scale = layer_scale
        self.scale_in = OffsetScale(in_dim) if self.layer_scale else nn.Identity()
        self.scale_out = OffsetScale(in_dim) if self.layer_scale else nn.Identity()

        self.drop_1 = nn.Dropout(moe_droprate_act) if moe_droprate_act is not None else nn.Dropout(moe_droprate)
        self.drop_2 = nn.Dropout(moe_droprate)
        
        self.freeze_moe = freeze_moe
        if self.freeze_moe:
            if self.glu:
                self.eva_fc1_g.weight.requires_grad = False
                self.eva_fc1_x.weight.requires_grad = False
                self.eva_fc2.weight.requires_grad = False
                self.eva_fc1_g.bias.requires_grad = False
                self.eva_fc1_x.bias.requires_grad = False
                self.eva_fc2.bias.requires_grad = False
            else:
                self.vit_fc1.weight.requires_grad = False
                self.vit_fc2.weight.requires_grad = False
                self.vit_fc1.bias.requires_grad = False
                self.vit_fc2.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input with embed_dim={self.in_features} (dim=-1), but "
                f"found {x.size(-1)}"
            )
        elif x.size(1) != self.num_experts:
            raise ValueError(
                f"Expected input with num_experts={self.num_experts} (dim=1), but "
                f"found {x.size(1)}"
            )
        # NOTE: 'd1' and 'd2' are both equal to 'embed_dim'. But for 'einsum' to
        # work correctly, we have to give them different names.
        x = self.scale_in(x)

        if self.glu:
            x_gate = self.eva_fc1_g(x)
            x = self.eva_fc1_x(x)
            x = self.act_fn(x_gate) * x
            x = self.drop_1(x)
            x = self.eva_norm(x)
            x = self.eva_fc2(x)
        else:
            x = self.vit_fc1(x)
            x = self.act_fn(x)
            x = self.drop_1(x)
            x = self.norm(x)
            x = self.vit_fc2(x)

        x = self.drop_2(x)
        x = self.scale_out(x)
        return x