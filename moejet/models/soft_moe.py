import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fast_softmoe_layer import MultiExpertLayer
from einops import rearrange
from typing import Callable, Optional, Tuple, Union
from timm.layers import DropPath
import torch.nn.functional as F
from timm.layers import trunc_normal_, lecun_normal_
from entmax import sparsemax, entmax15


# def l2norm(t, dim=-1):
#     return F.normalize(t, dim = dim)

# def l2norm(x):
#     return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

def l2norm(x, dim=-1, eps=1e-6):
    norm = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True))
    return x * (1 / (norm + eps))

def stable_softmax(x, dim):
    z = x - x.max(dim=dim, keepdim=True)[0]
    return torch.exp(z) / torch.exp(z).sum(dim=dim, keepdim=True)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x, norm_dim=-1):
        # return l2norm(x, norm_dim) * self.scale * self.gamma
        return l2norm(x, norm_dim) * self.gamma

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def normal_noise(t):
    # noise = torch.normal(mean=0, std=1, size=t.shape[-2:],requires_grad=False).to(t.device)
    noise = torch.normal(mean=0, std=1, size=t.shape, device=t.device)
    return noise

class NormalNoiseGenerator(nn.Module):
    """Generates a random noisy mask for logits tensor.

    All noise is generated from a normal distribution :math:`(0, 1 / E^2)`, where
    `E = the number of experts`.

    Args:
        num_experts (int): The number of experts.
    """

    def __init__(self, num_experts: int):
        super(NormalNoiseGenerator, self).__init__()
        self.num_experts = num_experts
        self.loc = 0.0
        self.scale = 1.0 / num_experts**2

    def forward(self, inputs: torch.Tensor):
        device = inputs.device
        normal_dist = torch.distributions.normal.Normal(
            loc=torch.tensor(self.loc, device=device),
            scale=torch.tensor(self.scale, device=device)
        )
        noisy = normal_dist.rsample(inputs.shape)
        return inputs + noisy


class LearnableNormalNoiseGenerator(nn.Module):
    """Generates a random noisy mask for logits tensor.

    All noise is generated from a normal distribution :math:`(0, 1 / E^2)`, where
    `E = the number of experts`.

    Args:
        num_experts (int): The number of experts.
    """

    def __init__(self, num_experts: int):
        super(LearnableNormalNoiseGenerator, self).__init__()
        self.num_experts = num_experts
        self.loc_min, self.loc_max = -0.1, 0.1
        self.scale_min, self.scale_max = 1e-12, 0.1

        # Initialize parameters
        initial_loc_value = 0.0
        initial_scale_value = 1.0 / num_experts**2
        self.loc = nn.Parameter(self._initialize_param(initial_loc_value, self.loc_min, self.loc_max))
        self.scale = nn.Parameter(self._initialize_param(initial_scale_value, self.scale_min, self.scale_max))

    def forward(self, inputs: torch.Tensor):
        loc = self._apply_constraints(self.loc, self.loc_min, self.loc_max)
        scale = self._apply_constraints(self.scale, self.scale_min, self.scale_max)
        normal_dist = torch.distributions.normal.Normal(loc=loc, scale=scale)
        noisy = normal_dist.rsample(inputs.shape)
        return inputs + noisy

    def _apply_constraints(self, param, min_val, max_val):
        # Apply constraints using a sigmoid function to keep values within [min_val, max_val]
        return min_val + (max_val - min_val) * torch.sigmoid(param)

    def _initialize_param(self, target_value, min_val, max_val):
        # Calculate the initial parameter value that will result in target_value after applying constraints
        target_scaled = (target_value - min_val) / (max_val - min_val)
        initial_param = torch.log(torch.tensor(target_scaled / (1.0 - target_scaled)))
        return initial_param


class UniformNoiseGenerator(nn.Module):
    """Generates Uniform noise for logits tensor."""

    def __init__(self, num_experts: int):
        super(UniformNoiseGenerator, self).__init__()
        self.num_experts = num_experts
        self.eps = 1e-2

    def forward(self, inputs: torch.Tensor):
        device = inputs.device
        uniform_dist = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 + self.eps, device=device),
            high=torch.tensor(1.0 - self.eps, device=device)
        )
        noisy = uniform_dist.rsample(inputs.shape)
        return inputs * noisy

class OffsetScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = x * self.gamma + self.beta
        return out


def softmax(x: torch.Tensor, dim) -> torch.Tensor:
    """
    Compute the softmax along the specified dimensions.
    This function adds the option to specify multiple dimensions

    Args:
        x (torch.Tensor): Input tensor.
        dims (int or tuple[int]): The dimension or list of dimensions along which the softmax probabilities are computed.

    Returns:
        torch.Tensor: Output tensor containing softmax probabilities along the specified dimensions.
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


def adjust_query_length(queries, new_length):
    """
    Adjust the length of queries to new_length using interpolation.
    
    :param queries: Input queries tensor of shape (N, query_len, embed_size)
    :param new_length: The desired length of the queries
    :return: New queries of shape (N, new_length, embed_size)
    """
    N, query_len, embed_size = queries.shape
    # Change the shape to (N, embed_size, query_len) for interpolation
    queries = queries.transpose(1, 2)
    # Interpolate along the last dimension (original query_len)
    queries_interpolated = F.interpolate(queries, size=new_length, mode='linear', align_corners=False)
    # Transpose back to (N, new_length, embed_size)
    queries_interpolated = queries_interpolated.transpose(1, 2)
    return queries_interpolated


class SoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        add_noise: bool = False,
        noise_mult: float = 1.0,
        moe_droprate: float = 0.0,
        moe_drop_path_rate: float = 0.0,
        moe_logits_drop: float = 0.0,
        compress_ratio: float = 1.0,
        phi: nn.Parameter = None,
        key_proj: bool = False,
        query_proj: bool = False,
        slot_layernorm: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.add_noise = add_noise
        self.noise_mult = noise_mult

        # Initialize phi and normalization scaling factor
        # self.register_parameter('phi' , nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim)))
        if phi is None:
            self.phi = nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim))
            # Initialize phi using LeCun normal initialization
            # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
            nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)
        else:
            self.phi = phi

        if slot_layernorm:
            self.norm = nn.Identity()
            self.slot_norm = nn.LayerNorm(dim)
        else:
            self.norm = l2norm
            self.slot_norm = l2norm
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        self.key_proj = key_proj
        if self.key_proj:
            self.phi_key_proj = OffsetScale(dim)
        else:
            self.phi_key_proj = nn.Identity()

        self.query_proj = query_proj
        if self.query_proj:
            self.phi_query_proj = OffsetScale(dim)
        else:
            self.phi_query_proj = nn.Identity()

        # Create a list of expert networks
        # self.experts = nn.ModuleList(
        #     [layer(**layer_kwargs) for _ in range(num_experts)]
        # )
        self.experts = MultiExpertLayer(
            in_dim=dim, 
            hidden_dim=int(4* compress_ratio * dim), 
            num_experts=num_experts, 
            moe_droprate=moe_droprate,
            layer_scale=kwargs.get('layer_scale', False),
            freeze_moe=kwargs.get('freeze_moe', False),
            )
        
        self.moe_logits_drop = moe_logits_drop
        self.expert_drop = DropPath(moe_drop_path_rate) if moe_drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"

        if len(x.shape) == 4:
            is_conv = True
            x = rearrange(x, 'b h w d -> b (h w) d')
        else:
            is_conv = False
            assert (len(x.shape) == 3), f"Input expected to have 3 or 4 dimensions but has {len(x.shape)}"

        query = self.phi

        # Normalize input and phi
        key = self.norm(self.phi_key_proj(x))
        query = self.slot_norm(self.phi_query_proj(query))
        query = query * self.scale
            # if x_norm.numel() < phi.numel():
            #     x_norm = x_norm * self.scale
            # else:
            #     phi = phi * self.scale

        # Compute dispatch and combine weights
        logits = torch.einsum("b n d, e s d->b n e s", key, query)

        # noised dispatch and combine gate logits, with annealing if needed

        if self.add_noise:
            noise = normal_noise(logits) * self.noise_mult
            logits = logits + noise

        if self.moe_logits_drop > 0.0:
            mask = torch.ones_like(logits) * self.moe_logits_drop
            mask = torch.bernoulli(mask) * -1e12
            logits = logits + mask

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # Compute input slots as weighted average of input tokens using dispatch weights
        slots = torch.einsum('b n d, b n e s -> b e s d', x, dispatch_weights)

        # Apply expert to corresponding slots
        # out = torch.stack(
        #     [f_i(slots[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        # )
        out = self.experts(slots)
        # Compute output tokens as weighted average of output slots using combine weights
        out = rearrange(out, ' b e s d -> b (e s) d')
        out = self.expert_drop(out)
        out = torch.einsum('b s d, b n s -> b n d', out, combine_weights)
        if is_conv:
            out = rearrange(out, 'b (h w) d -> b h w d', h=int(out.shape[1]**0.5))

        return out


class DualPathSoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        add_noise: bool = False,
        noise_mult: float = 1.0,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        moe_logits_drop: float = 0.0,
        moe_drop_path_rate: float = 0.0,
        compress_ratio: float = 1.0,
        phi = None,
        key_proj: bool = False,
        query_proj: bool = False,
        input_as_phi: bool = False,
        slot_layernorm: bool = False,
        multi_head: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.add_noise = add_noise
        self.noise_mult = noise_mult

        self.univ_factor = kwargs.get('univ_factor', 1/4)
        self.N_core = kwargs.get('core_experts', int(self.num_experts / 2))
        self.N_univ = kwargs.get('univ_experts', int(self.N_core / self.univ_factor))
        logit_scale = kwargs.get('logit_scale', 1.0)

        self.multi_head = multi_head
        if self.multi_head:
            self.num_heads = kwargs.get('num_heads', 3)
            self.head_dim = dim // self.num_heads

        # Initialize phi and normalization scaling factor
        # self.register_parameter('phi' , nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim)))
        self.input_as_phi = input_as_phi
        self.num_experts = self.N_core + self.N_univ
        self.slots_per_expert = slots_per_expert
        if phi is None and not input_as_phi:
            
            # self.phi = nn.Parameter(torch.zeros((self.N_core + self.N_univ), slots_per_expert, dim))
            # trunc_normal_(self.phi, std=0.02)
            query_init = self.initialize_query((self.N_core + self.N_univ), slots_per_expert, dim, pattern_type='orthogonal')
            self.phi = nn.Parameter(query_init)

        elif not input_as_phi:
            self.phi = phi

        if slot_layernorm:
            # self.norm = nn.Identity()
            self.norm = l2norm
            self.slot_norm = nn.LayerNorm(dim) if not self.multi_head else nn.LayerNorm(self.head_dim)
        else:
            self.norm = l2norm
            self.slot_norm = nn.Identity()
        # self.scale = nn.Parameter(torch.tensor(1.0) * logit_scale)
        self.scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5) * logit_scale),
            nn.Parameter(torch.tensor(0.5) * logit_scale)
        ])
        
        self.key_proj = key_proj
        if self.key_proj:
            self.phi_key_proj = OffsetScale(dim) if not self.multi_head else OffsetScale(self.head_dim)
            # self.phi_key_proj = nn.Linear(dim, dim)

        else:
            self.phi_key_proj = nn.Identity()

        self.query_proj = query_proj
        if self.query_proj:
            self.phi_query_proj = OffsetScale(dim)
            # self.phi_query_proj = nn.Linear(dim, dim)
        else:
            self.phi_query_proj = nn.Identity()

        self.moe_logits_drop = moe_logits_drop
        # self.moe_logits_drop = nn.Dropout(moe_logits_drop) if moe_logits_drop > 0.0 else nn.Identity()
        self.expert_drop = DropPath(moe_drop_path_rate) if moe_drop_path_rate > 0. else nn.Identity()
        self.core_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * compress_ratio * dim), num_experts=self.N_core, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act)
        self.univ_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * self.univ_factor * compress_ratio * dim), num_experts=self.N_univ, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act)

        # self.noise_generator = LearnableNormalNoiseGenerator(self.num_experts)

    def initialize_query(self, num_experts, slots_per_expert, d_model, pattern_type='orthogonal'):
        query_init = torch.randn(num_experts, slots_per_expert, d_model)
        if pattern_type == 'orthogonal':
            nn.init.orthogonal_(query_init)
        elif pattern_type == 'trunc_normal':
            trunc_normal_(query_init, mean=0.0, std=0.02)  # 截断正态分布初始化
        else:
            raise ValueError(f"Unsupported pattern_type: {pattern_type}")

        # pos_encoding = torch.arange(0, d_model).unsqueeze(0).repeat(num_experts, 1).float()
        # query_init += 0.01 * pos_encoding

        return query_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"

        if len(x.shape) == 4:
            is_conv = True
            x = rearrange(x, 'b h w d -> b (h w) d')
        else:
            is_conv = False
            assert (len(x.shape) == 3), f"Input expected to have 3 or 4 dimensions but has {len(x.shape)}"

        if not self.input_as_phi:
            query = self.phi
        else:
            query = adjust_query_length(x, self.num_experts*self.slots_per_expert)
            query = rearrange(query, 'b (e s) d -> b e s d', e=int(self.num_experts))

        # Normalize input and phi
        if self.multi_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
            query = rearrange(query, 'e s (h d) -> h e s d', h=self.num_heads)

        key = self.norm(self.phi_key_proj(x))
        query = self.slot_norm(self.phi_query_proj(query))
        query = l2norm(query)
        # query = query * self.scale

        # Compute dispatch and combine weights
        if self.multi_head:
            logits = torch.einsum("b h n d, h e s d -> b h n e s", key, query)
        else: 
            if not self.input_as_phi:
                logits = torch.einsum("b n d, e s d->b n e s", key, query)
            else:
                logits = torch.einsum("b n d, b e s d->b n e s", key, query)
                logits = logits / (self.dim ** 0.5)

        # noised dispatch and combine gate logits, with annealing if needed
        dispatch_logits = logits / self.scales[0]
        combine_logits = logits / self.scales[1]
        
        if self.add_noise:
            noise = normal_noise(logits) * self.noise_mult
            # logits = logits + noise
            dispatch_logits = dispatch_logits + noise
            combine_logits = combine_logits + noise
            # logits = self.noise_generator(logits)

        if self.moe_logits_drop > 0.0:
            mask = torch.ones_like(logits) * self.moe_logits_drop
            mask = torch.bernoulli(mask) * -1e12
            logits = logits + mask
        
        # dispatch_weights = dispatch_logits.softmax(dim = -3)
        # combine_weights = combine_logits.flatten(start_dim=-2).softmax(dim = -1)

        dispatch_weights = softmax(dispatch_logits, dim=-3)
        combine_weights = entmax15(
                combine_logits.flatten(start_dim=-2), dim=-1)

        # dispatch_weights = self.moe_logits_drop(dispatch_weights)
        # combine_weights = self.moe_logits_drop(combine_weights)

        slots = torch.einsum('... n d, ... n e s -> ... e s d', x, dispatch_weights)

        if self.multi_head:
            slots = rearrange(slots, 'b h e s d -> b e s (h d)')

        core_slots = slots[:, :self.N_core, :, :]
        univ_slots = slots[:, self.N_core:, :, :]

        core_out = self.core_experts(core_slots)
        univ_out = self.univ_experts(univ_slots)

        out = torch.cat((core_out, univ_out), dim=1)

        # Compute output tokens as weighted average of output slots using combine weights
        if self.multi_head:
            out = rearrange(out, ' b e s (h d) -> b h (e s) d', h=self.num_heads)
        else:
            out = rearrange(out, ' b e s d -> b (e s) d')

        out = self.expert_drop(out)
        out = torch.einsum('... s d, ... n s -> ... n d', out, combine_weights)

        if self.multi_head:
            out = rearrange(out, 'b h n d -> b n (h d)', h=self.num_heads)

        if is_conv:
            out = rearrange(out, 'b (h w) d -> b h w d', h=int(out.shape[1]**0.5))

        return out
