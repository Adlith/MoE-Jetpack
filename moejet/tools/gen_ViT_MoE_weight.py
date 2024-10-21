# %%
import os
import sys
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import functools
from torch.utils.data import DataLoader, Subset
import numpy as np
from einops import einsum, rearrange
from community import community_louvain
from sklearn.cluster import SpectralClustering
import pymetis
import networkx as nx
from mmpretrain.models import build_classifier
from mmengine.config import Config
import re
from pathlib import Path

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent


path = current_dir.parents[1]
sys.path.append(path)

torch.random.manual_seed(3407)
DEVICE = 'cuda:6' if torch.cuda.is_available() else 'cpu'
DEVICE_ID = [6,7]

data_path = path / 'data' / 'imagenet'
val_batch_size = 256

save_path = path / 'moejet' / 'weights' / 'gen_weight'
save_path.mkdir(parents=True, exist_ok=True)

vit_s_path = path / 'moejet' / 'weights' / 'vit_small_patch16_224.augreg_in21k.pth'
vit_t_path = path / 'moejet' / 'weights' / 'vit_tiny_patch16_224.augreg_in21k.pth'


# %%
def prepare_imagenet(imagenet_root, val_batch_size, num_workers, model):
    data_config = timm.data.resolve_model_data_config(model)
    val_dst = ImageFolder(  # 1281167
        # os.path.join(imagenet_root, 'val'),
        os.path.join(imagenet_root, 'train'),
        transform = timm.data.create_transform(**data_config, is_training=False)
    )

    indices = torch.arange(128 * 300)  # 1000
    val_dst = Subset(val_dst, indices)

    val_loader = DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return val_loader

# %%
def batch_mean(data, batch_size=100):
    num_batches = len(data) // batch_size + (0 if len(data) % batch_size == 0 else 1)
    means = []
    for i in range(num_batches):
        batch = data[i * batch_size: (i + 1) * batch_size]
        means.append(np.mean(batch, axis=0))
    overall_mean = np.mean(np.stack(means), axis=0)
    return overall_mean


def feature_analysis(model, layer_indices, val_loader, position='mlp.act', threshold=0.0):
    
    def _hook_function(module, input, output, idx):
        if if_input:
            data = input[0]
        else:
            data = output

        with torch.no_grad():
            activation_map = (data > threshold).float()  # Shape: (b, n, d)
            co_act = einsum(activation_map, activation_map, 'b n d1, b n d2-> b d1 d2')  # Shape: (b, d, d)
            co_act = co_act.sum(dim=0).cpu().numpy()  # Shape: (d, d)
            

            l1_norm = data.norm(p=1, dim=(0, 1)).cpu().numpy()
            l2_norm = data.norm(p=2, dim=(0, 1)).cpu().numpy()
            activation_pattern = (data > threshold).float().sum(dim=(0, 1)).cpu().numpy()

            # graph = nx.from_numpy_array(co_act)
            # partition = apply_graph_partitioning(graph, method='metis', hypter=32)
            # community_sizes = get_community_sizes(partition)       

            activation_stats[idx]['L1'].append(l1_norm)
            activation_stats[idx]['L2'].append(l2_norm)
            activation_stats[idx]['act_patrn'].append(activation_pattern)
            activation_stats[idx]['co_act'].append(co_act)
            
    
    model = model.to(DEVICE)
    model = nn.DataParallel(model, device_ids=DEVICE_ID)
    model.eval()

    # 初始化存储结构
    activation_stats = {idx: {'L1': [], 'L2': [], 'act_patrn': [], 'co_act': []} for idx in layer_indices}

    hooks = []
    if_input = False
    for idx in layer_indices:
        # 根据idx确定是注册在mlp.act还是mlp.fc2
        if position == 'mlp.act':
            target_layer = model.module.blocks[idx].mlp.act
        elif position == 'mlp.fc2':
            target_layer = model.module.blocks[idx].mlp.fc2
        elif position == 'attn.proj':  # proj input
            target_layer = model.module.blocks[idx].attn.proj
            if_input = True
        elif position == 'norm2':
            target_layer = model.module.blocks[idx].norm2
        else:
            raise ValueError('Invalid position argument')
        hook = target_layer.register_forward_hook(functools.partial(_hook_function, idx=idx))
        hooks.append(hook)

    with torch.no_grad():
        for _, (images, labels) in enumerate(tqdm(val_loader, desc="val images")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _ = model(images)

    for hook in hooks:
        hook.remove()

    # for idx, stats in activation_stats.items():
    #     for key in stats:
    #         stats[key] = np.mean(stats[key], axis=0)
    
    for idx, stats in tqdm(activation_stats.items(), desc="compute mean"):
        for key in stats:
            stats[key] = batch_mean(stats[key], batch_size=256)

    # activation_stats = normalize_stats(activation_stats)

    return activation_stats


def normalize_stats(activation_stats):
    for stat_key in ['L1', 'L2', 'act_patrn']:
        # 收集所有层的当前统计指标
        all_layers_stats = np.concatenate([activation_stats[idx][stat_key] for idx in activation_stats], axis=0)
        
        # 计算全局平均值和标准差
        global_mean = np.mean(all_layers_stats, axis=0)
        global_std = np.std(all_layers_stats, axis=0)

        # 避免除以零的情况，添加一个小的epsilon
        epsilon = 1e-8
        global_std = np.where(global_std > 0, global_std, epsilon)

        # 对每一层的统计指标应用Z得分归一化
        for idx in activation_stats:
            activation_stats[idx][stat_key] = (activation_stats[idx][stat_key] - global_mean) / global_std

    return activation_stats

# %%
def apply_graph_partitioning(graph, method='metis', hypter=30):
    if method == 'spectral':
        # 光谱聚类方法参数
        num_clusters = hypter
        adjacency_matrix = nx.adjacency_matrix(graph).toarray()
        sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', n_init=100)
        labels = sc.fit_predict(adjacency_matrix)
        partition = {node: label for node, label in enumerate(labels)}
    
    elif method == 'metis':
        # METIS 方法参数
        num_partitions = hypter
        adjacency_list = [list(graph.neighbors(i)) for i in range(graph.number_of_nodes())]
        _, parts = pymetis.part_graph(num_partitions, adjacency=adjacency_list)
        partition = {node: part for node, part in enumerate(parts)}

    elif method == 'louvain':
        # Louvain 方法参数
        resolution = hypter
        partition = community_louvain.best_partition(graph, resolution=resolution)
        size = len(set(partition.values()))
        print(f"Resolution {resolution}: {size} communities detected")
    
    else:
        raise ValueError("Unsupported partitioning method")
    
    part_to_nodes = {}
    for node, part in partition.items():
        if part not in part_to_nodes:
            part_to_nodes[part] = []
        part_to_nodes[part].append(node)
    
    return partition, part_to_nodes

def get_community_sizes(partition):
    # Initialize a dictionary to hold the count of nodes in each community
    community_sizes = {}
    for node, community in partition.items():
        if community in community_sizes:
            community_sizes[community] += 1
        else:
            community_sizes[community] = 1
    for community, size in community_sizes.items():
        print(f"Community {community}: {size} nodes")
    return community_sizes

# %%
def generate_random_indices(total_size, subset_size, device):
    # 生成一个随机排列的索引子集
    perm = torch.randperm(total_size, device=device)
    indices = perm[:subset_size]
    return torch.sort(indices).values

def random_indices(num_experts, moe_blocks, total_size, subset_ratio, device, select_channel=False):
    # 为每个专家生成独立的随机索引并存储在字典中
    subset_size = int(total_size * subset_ratio)
    if select_channel:
        channel_indices = generate_random_indices(total_size, subset_size, device)

    layer_indices_dict = {}
    for j in range(12):
        if j in moe_blocks and num_experts > 1:
            per_expt_indices = {}
            for i in range(num_experts):
                indices = generate_random_indices(total_size, subset_size, device)
                per_expt_indices[i] = indices
            layer_indices_dict[j] = per_expt_indices
        else:
            indices = channel_indices if select_channel else generate_random_indices(total_size, subset_size, device)
            layer_indices_dict[j] = indices
    return layer_indices_dict
    
def generate_uniform_indices(total_size, subset_size, device):
    # 使用 torch.linspace 生成均匀间隔的浮点数，然后进行四舍五入并转换为整数
    indices = torch.round(torch.linspace(0, total_size - 1, subset_size, device=device)).long()
    return torch.sort(indices).values

def uniform_indices(num_experts, moe_blocks, total_size, subset_ratio, device):
    subset_size = int(total_size * subset_ratio)
    indices = generate_uniform_indices(total_size, subset_size, device)
    layer_indices_dict = {}
    for j in range(12):
        if j in moe_blocks and num_experts > 1:
            per_expt_indices = {}
            for i in range(num_experts):
                per_expt_indices[i] = indices
            layer_indices_dict[j] = per_expt_indices
        else:
            layer_indices_dict[j] = indices
    return layer_indices_dict
    
def selected_indices_Pruning(analysis_feature, num_experts, moe_blocks, total_size, subset_ratio, type='L1', select_channel=False):
    assert type in ['L1', 'L2', 'act_patrn']
    
    subset_size = int(total_size * subset_ratio)
    total_pruning_metric = torch.zeros(total_size)
    layer_probs = {}
    for layer_name, layer_stats in analysis_feature.items():
        total_pruning_metric += layer_stats[type]
        layer_prob = layer_stats[type] / layer_stats[type].sum()  # 计算该层的概率
        layer_probs[layer_name] = torch.tensor(layer_prob)

    # 计算概率分布
    if select_channel:
        # probs = total_pruning_metric / total_pruning_metric.sum()
        # channel_indices = torch.multinomial(probs, subset_size, replacement=False)

        _, channel_indices = torch.topk(total_pruning_metric, subset_size, largest=True)

    layer_indices_dict = {}
    for j in range(12):
        if j in moe_blocks and num_experts > 1:
            per_expt_indices = {}
            for i in range(num_experts):
                indices = torch.multinomial(layer_probs[j], subset_size, replacement=False)
                per_expt_indices[i] = torch.sort(indices).values
            layer_indices_dict[j] = per_expt_indices
        else:
            indices = channel_indices if select_channel else \
                torch.multinomial(layer_probs[j], subset_size, replacement=False)
            layer_indices_dict[j] = torch.sort(indices).values
    return layer_indices_dict
    

def generate_graph_prob(co_act_matrix, subset_size, clusters):
    graph = nx.from_numpy_array(co_act_matrix)
    _, part_to_nodes = apply_graph_partitioning(graph, method='metis', hypter=clusters)
    
    coactivation_sums = {}
    for part, nodes in part_to_nodes.items():
        # Calculate the sum of all co-activations within this subgraph
        coactivation_sum = sum(co_act_matrix[i, j] for i in nodes for j in nodes if i != j)
        coactivation_sums[part] = coactivation_sum
        
    subgraph_ids, coactivations = zip(*coactivation_sums.items())
    probabilities = torch.tensor(coactivations, dtype=torch.float) / sum(coactivations)
    return probabilities, part_to_nodes, subgraph_ids


def selected_indices_Graph(analysis_feature, num_experts, moe_blocks, total_size, subset_ratio, type='co_act', select_channel=True):
    assert type in ['co_act']
    
    subset_size = int(total_size * subset_ratio)
    clusters = 32
    
    example_layer = next(iter(analysis_feature.values()))
    matrix_size = example_layer[type].shape
    total_co_act = np.zeros(matrix_size)
    
    layer_co_act = {}
    for layer_name, layer_stats in analysis_feature.items():
        co_act_matrix = layer_stats[type]
        np.fill_diagonal(co_act_matrix, 0)
        
        total_co_act += co_act_matrix
        layer_co_act[layer_name] = co_act_matrix
    
    # 计算概率分布
    if select_channel:

        probabilities, part_to_nodes, subgraph_ids = generate_graph_prob(total_co_act, subset_size, clusters)
        sampled_indices = torch.multinomial(probabilities, (clusters//2+1), replacement=False)
        selected_nodes = [node for i in sampled_indices for node in part_to_nodes[subgraph_ids[i]]]
        channel_indices = torch.tensor(selected_nodes[:subset_size])

    layer_indices_dict = {}
    for j in range(12):
        if j in moe_blocks and num_experts > 1:
            
            per_expt_indices = {}
            probabilities, part_to_nodes, subgraph_ids = generate_graph_prob(layer_co_act[j], subset_size, clusters)
            
            for i in tqdm(range(num_experts), desc="Processing experts"):
                sampled_indices = torch.multinomial(probabilities, (clusters//2+1), replacement=False)
                selected_nodes = [node for s_i in sampled_indices for node in part_to_nodes[subgraph_ids[s_i]]]
                indices = torch.tensor(selected_nodes[:subset_size])

                per_expt_indices[i] = torch.sort(indices).values
            layer_indices_dict[j] = per_expt_indices
        else:
            if select_channel:
                indices = channel_indices
            else:
                probabilities, part_to_nodes, subgraph_ids = generate_graph_prob(layer_co_act[j], subset_size, clusters)
                sampled_indices = torch.multinomial(probabilities, (clusters//2+1), replacement=False)
                selected_nodes = [node for s_i in sampled_indices for node in part_to_nodes[subgraph_ids[s_i]]]
                indices = torch.tensor(selected_nodes[:subset_size])
                
            layer_indices_dict[j] = torch.sort(indices).values
    return layer_indices_dict


def generate_expert_indices(model, MoE_dict, num_experts, moe_blocks, mlp_act, attn_proj, norm2, select_type):
    teacher_weights = model.state_dict()
    s_channel = MoE_dict['blocks.0.norm1.weight'].shape[0]

    device = teacher_weights['pos_embed'].device
    t_channel = teacher_weights['blocks.0.norm1.weight'].shape[0]
    t_attn = teacher_weights['blocks.0.attn.proj.weight'].shape[1]
    t_mlp = teacher_weights['blocks.0.mlp.fc2.weight'].shape[1]

    s_channel_ratio = s_channel / t_channel

    selected_idx = {}
    if select_type == 'random':
        selected_idx['moe_channel_idx'] = random_indices(num_experts, moe_blocks, t_channel, s_channel_ratio, device)
        selected_idx['moe_attn_idx']  = random_indices(num_experts, moe_blocks, t_attn, s_channel_ratio, device)
        selected_idx['moe_mlp_idx'] = random_indices(num_experts, moe_blocks, t_mlp, s_channel_ratio, device)

        selected_idx['dense_channel_idx'] = random_indices(1, moe_blocks, t_channel, s_channel_ratio, device, select_channel=True)
        selected_idx['dense_attn_idx'] = random_indices(1, moe_blocks, t_attn, s_channel_ratio, device)
        selected_idx['dense_mlp_idx'] = random_indices(1, moe_blocks, t_mlp, s_channel_ratio, device)
    elif select_type == 'uniform':
        selected_idx['moe_channel_idx'] = uniform_indices(num_experts, moe_blocks, t_channel, s_channel_ratio, device)
        selected_idx['moe_attn_idx'] = uniform_indices(num_experts, moe_blocks, t_attn, s_channel_ratio, device)
        selected_idx['moe_mlp_idx'] = uniform_indices(num_experts, moe_blocks, t_mlp, s_channel_ratio, device)

        selected_idx['dense_channel_idx'] = uniform_indices(1, moe_blocks, t_channel, s_channel_ratio, device)
        selected_idx['dense_attn_idx'] = uniform_indices(1, moe_blocks, t_attn, s_channel_ratio, device)
        selected_idx['dense_mlp_idx'] = uniform_indices(1, moe_blocks, t_mlp, s_channel_ratio, device)
    elif select_type == 'L1' or select_type == 'L2' or select_type == 'act_patrn':
        selected_idx['moe_channel_idx'] = selected_indices_Pruning(norm2, num_experts, moe_blocks, t_channel, s_channel_ratio, type=select_type)
        selected_idx['moe_attn_idx'] = selected_indices_Pruning(attn_proj, num_experts, moe_blocks, t_attn, s_channel_ratio, type=select_type)
        selected_idx['moe_mlp_idx'] = selected_indices_Pruning(mlp_act, num_experts, moe_blocks, t_mlp, s_channel_ratio, type=select_type)

        selected_idx['dense_channel_idx'] = selected_indices_Pruning(norm2, 1, moe_blocks, t_channel, s_channel_ratio, type=select_type, select_channel=True)
        selected_idx['dense_attn_idx'] = selected_indices_Pruning(attn_proj, 1, moe_blocks, t_attn, s_channel_ratio, type=select_type)
        selected_idx['dense_mlp_idx'] = selected_indices_Pruning(mlp_act, 1, moe_blocks, t_mlp, s_channel_ratio, type=select_type)
    elif select_type == 'co_act':
        selected_idx['moe_channel_idx'] = selected_indices_Graph(norm2, num_experts, moe_blocks, t_channel, s_channel_ratio, type=select_type)
        selected_idx['moe_attn_idx'] = selected_indices_Graph(attn_proj, num_experts, moe_blocks, t_attn, s_channel_ratio, type=select_type)
        selected_idx['moe_mlp_idx'] = selected_indices_Graph(mlp_act, num_experts, moe_blocks, t_mlp, s_channel_ratio, type=select_type)

        selected_idx['dense_channel_idx'] = selected_indices_Graph(norm2, 1, moe_blocks, t_channel, s_channel_ratio, type=select_type, select_channel=True)
        selected_idx['dense_attn_idx'] = selected_indices_Graph(attn_proj, 1, moe_blocks, t_attn, s_channel_ratio, type=select_type)
        selected_idx['dense_mlp_idx'] = selected_indices_Graph(mlp_act, 1, moe_blocks, t_mlp, s_channel_ratio, type=select_type)
    else:
        raise ValueError("Unsupported selection type")
        

    return selected_idx

# %%
def expand_indices(indices, dim, times):
    expanded_indices = []
    # 初始化扩展列表，首先添加所有原始索引
    expanded_indices.extend(indices)
    
    # 为每个原始索引添加扩展
    for t in range(1, times):
        expanded_indices.extend([index + t * dim for index in indices])
    
    return torch.stack(expanded_indices)

def process_key(key, prefix='experts.'):
    key = re.sub(r'(blocks\.\d+\.)', r'\1' + prefix, key)
    return key

def process_key_mlp(key, prefix='experts.'):
    key = re.sub(r'(blocks\.\d+\.+mlp\.)', r'\1' + prefix, key)
    return key

def get_selected_weight(
        teacher_model, 
        is_glu, 
        dense_channel_idx, dense_attn_idx, dense_mlp_idx,
        moe_channel_idx, moe_attn_idx, moe_mlp_idx,
        moe_blocks, num_experts=196,
        select_unit='block',
    ):
    
    #  select_type: [random, uniform], [l1norm, l2norm, act_pattern], co_act
    teacher_weights =  teacher_model.state_dict()
    t_attn = teacher_weights['blocks.0.attn.proj.weight'].shape[1]
    
    student_weights = collections.OrderedDict()
    prefix = 'experts.'
    for key in tqdm(teacher_weights.keys(), desc="selecting weights"):
        # We don't perform weight selection on classification head by default. Remove this constraint if target dataset is the same as teacher's.
        if "head" in key:
            continue
        ws = teacher_weights[key].clone().cpu()
        if ('cls_token' in key) or ('pos_embed' in key) or \
            ('fc_norm' in key) or ('norm' in key and 'block' not in key):
            
            ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[0])
            student_weights[key] = ws

        elif 'patch_embed' in key:

            ws = torch.index_select(ws, dim=0, index=dense_channel_idx[0])
            student_weights[key] = ws

        elif "blocks" in key:
            layer_idx = int(key.split('.')[1])
            if layer_idx not in moe_blocks:
                if 'attn.qkv.weight' in key:
                    ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[layer_idx])
                    ws = torch.index_select(ws, dim=0, index=expand_indices(dense_attn_idx[layer_idx], t_attn, 3))
                elif 'attn.qkv.bias' in key:
                    ws = torch.index_select(ws, dim=0, index=expand_indices(dense_attn_idx[layer_idx], t_attn, 3))
                    
                elif ('attn.q_proj.weight' in key) or ('attn.k_proj.weight' in key) or ('attn.v_proj.weight' in key):
                    ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[layer_idx])
                    ws = torch.index_select(ws, dim=0, index=dense_attn_idx[layer_idx])
                elif ('attn.q_proj.bias' in key) or ('attn.k_proj.bias' in key) or ('attn.v_proj.bias' in key) or ('attn.norm' in key):
                    ws = torch.index_select(ws, dim=0, index=dense_attn_idx[layer_idx])
                    
                elif 'attn.proj.weight' in key:
                    ws = torch.index_select(ws, dim=0, index=dense_channel_idx[layer_idx])
                    ws = torch.index_select(ws, dim=-1, index=dense_attn_idx[layer_idx])
                elif 'attn.proj.bias' in key:
                    ws = torch.index_select(ws, dim=0, index=dense_channel_idx[layer_idx])
                elif 'mlp.fc1.weight' in key:
                    ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[layer_idx])
                    ws = torch.index_select(ws, dim=0, index=dense_mlp_idx[layer_idx])
                elif 'mlp.fc1.bias' in key:
                    ws = torch.index_select(ws, dim=0, index=dense_mlp_idx[layer_idx])

                elif ('mlp.fc1_g.weight' in key) or ('mlp.fc1_x.weight' in key):
                    ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[layer_idx])
                    ws = torch.index_select(ws, dim=0, index=dense_mlp_idx[layer_idx])
                elif ('mlp.fc1_g.bias' in key) or ('mlp.fc1_x.bias' in key) or ('mlp.norm' in key):
                    ws = torch.index_select(ws, dim=0, index=dense_mlp_idx[layer_idx])
                    
                elif 'mlp.fc2.weight' in key:
                    ws = torch.index_select(ws, dim=0, index=dense_channel_idx[layer_idx])
                    ws = torch.index_select(ws, dim=-1, index=dense_mlp_idx[layer_idx])
                elif 'mlp.fc2.bias' in key:
                    ws = torch.index_select(ws, dim=0, index=dense_channel_idx[layer_idx])
                elif 'norm1' in key or 'norm2' in key:
                    ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[layer_idx])
                else:
                    print(f"Unsupported weight selection key: {key}")
                    continue
                student_weights[key] = ws
            else:
                if select_unit == 'mlp':
                    if 'attn.qkv.weight' in key:
                        ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[layer_idx])
                        ws = torch.index_select(ws, dim=0, index=expand_indices(dense_attn_idx[layer_idx], t_attn, 3))
                    elif 'attn.qkv.bias' in key:
                        ws = torch.index_select(ws, dim=0, index=expand_indices(dense_attn_idx[layer_idx], t_attn, 3))
                    elif ('attn.q_proj.weight' in key) or ('attn.k_proj.weight' in key) or ('attn.v_proj.weight' in key):
                        ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[layer_idx])
                        ws = torch.index_select(ws, dim=0, index=dense_attn_idx[layer_idx])
                    elif ('attn.q_proj.bias' in key) or ('attn.k_proj.bias' in key) or ('attn.v_proj.bias' in key) or ('attn.norm' in key):
                        ws = torch.index_select(ws, dim=0, index=dense_attn_idx[layer_idx])
                    elif 'attn.proj.weight' in key:
                        ws = torch.index_select(ws, dim=0, index=dense_channel_idx[layer_idx])
                        ws = torch.index_select(ws, dim=-1, index=dense_attn_idx[layer_idx])
                    elif 'attn.proj.bias' in key:
                        ws = torch.index_select(ws, dim=0, index=dense_channel_idx[layer_idx])
                    elif 'mlp.fc1.weight' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=-1, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=0, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key_mlp(key, prefix)
                        key = key.replace('fc1', 'vit_fc1')
                    elif 'mlp.fc1.bias' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key_mlp(key, prefix)
                        key = key.replace('fc1', 'vit_fc1')
                    elif ('mlp.fc1_g.weight' in key) or ('mlp.fc1_x.weight' in key):
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=-1, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=0, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key_mlp(key, 'glu_'+prefix)
                        key = key.replace('fc1_g', 'eva_fc1_g')
                        key = key.replace('fc1_x', 'eva_fc1_x')
                    elif ('mlp.fc1_g.bias' in key) or ('mlp.fc1_x.bias' in key) or ('mlp.norm' in key):
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key_mlp(key, 'glu_'+prefix)
                        key = key.replace('fc1_g', 'eva_fc1_g')
                        key = key.replace('fc1_x', 'eva_fc1_x')
                        key = key.replace('norm', 'eva_norm')
                    elif 'mlp.fc2.weight' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=-1, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        if is_glu:
                            key = process_key_mlp(key, 'glu_'+prefix)
                            key = key.replace('fc2', 'eva_fc2')
                        else:
                            key = process_key_mlp(key, prefix)
                            key = key.replace('fc2', 'vit_fc2')
                    elif 'mlp.fc2.bias' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_channel_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        if is_glu:
                            key = process_key_mlp(key, 'glu_'+prefix)
                            key = key.replace('fc2', 'eva_fc2')
                        else:
                            key = process_key_mlp(key, prefix)
                            key = key.replace('fc2', 'vit_fc2')
                    elif 'norm1' in key or 'norm2' in key:
                        ws = torch.index_select(ws, dim=-1, index=dense_channel_idx[0])
                    else:
                        print(f"Unsupported weight selection key: {key}")
                        continue
                elif select_unit == 'block':
                    if 'attn.qkv.weight' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=-1, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=0, index=expand_indices(moe_attn_idx[layer_idx][e], t_attn, 3))
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                    elif 'attn.qkv.bias' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=expand_indices(moe_attn_idx[layer_idx][e], t_attn, 3))
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                    elif ('attn.q_proj.weight' in key) or ('attn.k_proj.weight' in key) or ('attn.v_proj.weight' in key):
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=-1, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=0, index=moe_attn_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                    elif ('attn.q_proj.bias' in key) or ('attn.k_proj.bias' in key) or ('attn.v_proj.bias' in key) or ('attn.norm' in key):
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_attn_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                    elif 'attn.proj.weight' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=-1, index=moe_attn_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                    elif 'attn.proj.bias' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_channel_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                    elif 'mlp.fc1.weight' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=-1, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=0, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                        key = key.replace('fc1', 'vit_fc1')
                    elif 'mlp.fc1.bias' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                        key = key.replace('fc1', 'vit_fc1')
                    elif ('mlp.fc1_g.weight' in key) or ('mlp.fc1_x.weight' in key):
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=-1, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=0, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                        key = key.replace('fc1_g', 'eva_fc1_g')
                        key = key.replace('fc1_x', 'eva_fc1_x')
                    elif ('mlp.fc1_g.bias' in key) or ('mlp.fc1_x.bias' in key) or ('mlp.norm' in key):
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                        key = key.replace('fc1_g', 'eva_fc1_g')
                        key = key.replace('fc1_x', 'eva_fc1_x')
                        key = key.replace('norm', 'eva_norm')
                    elif 'mlp.fc2.weight' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_channel_idx[layer_idx][e])
                            ws_tmp = torch.index_select(ws_tmp, dim=-1, index=moe_mlp_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                        if is_glu:
                            key = key.replace('fc2', 'eva_fc2')
                        else:
                            key = key.replace('fc2', 'vit_fc2')
                    elif 'mlp.fc2.bias' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=0, index=moe_channel_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                        if is_glu:
                            key = key.replace('fc2', 'eva_fc2')
                        else:
                            key = key.replace('fc2', 'vit_fc2')
                    elif 'norm1' in key or 'norm2' in key:
                        e_ws = []
                        for e in range(num_experts):
                            ws_tmp = torch.index_select(ws, dim=-1, index=moe_channel_idx[layer_idx][e])
                            e_ws.append(ws_tmp)
                        ws = torch.stack(e_ws, dim=0)
                        key = process_key(key, prefix)
                        key = key.replace('norm1', 'attn.norm1')
                        key = key.replace('norm2', 'mlp.norm2')
                    else:
                        print(f"Unsupported weight selection key: {key}")
                        continue
                else:
                    raise ValueError(f"Unsupported weight selection unit: {select_unit}")
                student_weights[key] = ws
        else:
            print(f"Unsupported weight type: {key}")

    return student_weights

def combine_eva_attn(state_dict, block_indices):
    # Helper function to concatenate weight or bias tensors
    bias_shape = state_dict['blocks.11.experts.attn.q_proj.bias'].shape
    def concatenate_tensors(key_base, suffix):
        tensors = []
        for key in [f'{key_base}.{proj}{suffix}' for proj in ['q_proj', 'k_proj', 'v_proj']]:
            tensor = state_dict.pop(key, None)
            if tensor is None:
                if suffix == '.bias':
                    tensor = torch.zeros(bias_shape)
                else:
                    raise KeyError(f"The key {key} is not found in the state_dict for block {block_idx}")
            tensors.append(tensor)
        return torch.cat(tensors, dim=1)  # Concatenating along the features dimension

    # Iterate through each block and combine projections
    for block_idx in block_indices:
        key_base = f'blocks.{block_idx}.experts.attn'
        # Combine the weight matrices
        state_dict[f'{key_base}.qkv.weight'] = concatenate_tensors(key_base, '.weight')
        # Combine the biases, if they exist
        state_dict[f'{key_base}.qkv.bias'] = concatenate_tensors(key_base, '.bias')

    return state_dict


def compare_keys_with_prefix(dict1, dict2, prefix):
    # Filter keys starting with the given prefix
    dict1_filtered_keys = {key for key in dict1 if key.startswith(prefix)}
    dict2_filtered_keys = {key for key in dict2 if key.startswith(prefix)}
    
    # Find keys that are unique to each dictionary
    unique_to_dict1 = dict1_filtered_keys - dict2_filtered_keys
    unique_to_dict2 = dict2_filtered_keys - dict1_filtered_keys

    # Find common keys and check if their order is the same
    common_keys_ordered_dict1 = [key for key in dict1 if key in dict1_filtered_keys & dict2_filtered_keys]
    common_keys_ordered_dict2 = [key for key in dict2 if key in dict1_filtered_keys & dict2_filtered_keys]
    
    order_mismatch = common_keys_ordered_dict1 != common_keys_ordered_dict2
    
    # Display the results
    print("======== Keys with prefix {} unique to dict1:".format(prefix))
    print(unique_to_dict1)
    print("======== Keys with prefix {} unique to dict2:".format(prefix))
    print(unique_to_dict2)
    print("Common keys with prefix {}:".format(prefix))
    print(dict1_filtered_keys & dict2_filtered_keys)
    print("Is the order of the common keys the same?")
    print(not order_mismatch)


def MoE_stitching(Dense_part, MoE_eva=None, MoE_vit=None, MoE_dict=None):
    print("------------ Comparing Dense ckpt with MoE")
    compare_keys_with_prefix(Dense_part, MoE_dict, '')
    if MoE_eva is not None:
        print("------------ Comparing eva ckpt with MoE")
        compare_keys_with_prefix(MoE_eva, MoE_dict, 'blocks.11.mlp')
    print("------------ Comparing vit ckpt with MoE")
    compare_keys_with_prefix(MoE_vit, MoE_dict, 'blocks.11.mlp')

    if MoE_dict is None:
        raise ValueError("MoE_dict must be provided.")
    
    # 遍历MoE_dict中的所有键
    for key in MoE_dict.keys():
        if 'experts' not in key:
            # 如果键不包含'experts'，从Dense_part复制值
            if key in Dense_part:
                if MoE_dict[key].shape == Dense_part[key].shape:
                    MoE_dict[key] = Dense_part[key]
                else:
                    print(f"Key D: {key} has shape {Dense_part[key].shape} in Dense_part and shape {MoE_dict[key].shape} in MoE_dict, skip.")
            else:
                print(f"Key D: {key} not found in Dense_part, skip.")
        else:
            # 如果键包含'experts'
            value_in_eva = key in MoE_eva if MoE_eva else False
            value_in_vit = key in MoE_vit if MoE_vit else False

            if value_in_eva and value_in_vit:
                # 如果键在MoE_eva和MoE_vit中都存在，则合并
                MoE_dict[key] = torch.cat((MoE_eva[key], MoE_vit[key]), dim=0)
            elif value_in_eva:
                # 如果键只在MoE_eva中存在
                MoE_dict[key] = MoE_eva[key]
            elif value_in_vit:
                # 如果键只在MoE_vit中存在
                MoE_dict[key] = MoE_vit[key]
            else:
                print(f"Key E: {key} not found in any of MoE_eva or MoE_vit.")

    return MoE_dict

# %%
def convert_vit_to_dualpath(model_dict, moe_blocks, core_experts, univ_experts, univ_factor, Dense_part=None):
    n_student_neuron = model_dict[f'blocks.{moe_blocks[0]}.mlp.experts.vit_fc1.bias'].shape[-1]
    device = model_dict['pos_embed'].device
    
    index = {}
    for i in moe_blocks:
        index[i] = generate_uniform_indices(n_student_neuron, int(n_student_neuron * univ_factor), device)

    dual_path_dict = collections.OrderedDict()
    for key, weight in model_dict.items():
        if 'vit_fc' in key and Dense_part is None:
            layer_idx = int(key.split('.')[1])

            core_weight = weight[:core_experts]
            core_key = key.replace('experts.', 'core_experts.')
            dual_path_dict[core_key] = core_weight

            univ_weight = weight[core_experts:]

            if 'vit_fc2.weight' in key:
                univ_weight = torch.index_select(univ_weight, dim=-1, index=index[layer_idx])
            elif ('vit_fc1.weight' in key) or ('vit_fc1.bias' in key):
                univ_weight = torch.index_select(univ_weight, dim=1, index=index[layer_idx])
            elif 'vit_fc2.bias' in key:
                univ_weight = univ_weight

            univ_key = key.replace('experts.', 'univ_experts.')
            dual_path_dict[univ_key] = univ_weight
        elif 'vit_fc' in key and Dense_part is not None:
            core_weight = weight[:core_experts]
            core_key = key.replace('experts.', 'core_experts.')
            dual_path_dict[core_key] = core_weight
            
            dense_key = key.replace('experts.', '').replace('vit_fc', 'fc')

            univ_weight = []
            for i in range(univ_experts):
                single_univ_weight = Dense_part[dense_key].clone()
                univ_index = generate_random_indices(n_student_neuron, int(n_student_neuron * univ_factor), device)
                if 'vit_fc2.weight' in key:
                    univ_weight.append(torch.index_select(single_univ_weight, dim=-1, index=univ_index))
                elif ('vit_fc1.weight' in key) or ('vit_fc1.bias' in key):
                    univ_weight.append(torch.index_select(single_univ_weight, dim=0, index=univ_index))
                elif 'vit_fc2.bias' in key:
                    univ_weight.append(single_univ_weight)
            univ_weight = torch.stack(univ_weight, dim=0)
            univ_key = key.replace('experts.', 'univ_experts.')
            dual_path_dict[univ_key] = univ_weight
        else:
            dual_path_dict[key] = weight
    
    return dual_path_dict

# %%
def main():
    
    moe_config_path = path / 'moejet/configs/timm/vit_tiny_dual_moe_timm_21k_ft.py'

    select_unit = 'mlp'  # block, mlp

    dataset_arg = (data_path, val_batch_size, 8)

    select_type = 'L2'  #  select_type: [random, uniform], [L1, L2, act_patrn], [co_act, kmeans]

    moe_cfg = Config.fromfile(moe_config_path)
    MoE = build_classifier(moe_cfg['model'])
    MoE_dict = MoE.state_dict()

    moe_blocks = moe_cfg.model.block_moe_layers if select_unit == 'block' else moe_cfg.model.moe_layers
    n_EXPs = moe_cfg.model.num_experts
    if_dualpath = 'dual' in moe_config_path.name
    if if_dualpath:
        c_EXPs = moe_cfg.model.core_experts
        u_EXPs = moe_cfg.model.univ_experts
        univ_factor = moe_cfg.model.univ_factor
        n_EXPs = c_EXPs + u_EXPs

    model_configs = [
        # {'name': 'vit_tiny_patch16_224.augreg_in21k', 'path': vit_t_path, 'dict_key': 'vit_t'},  # 
        {'name': 'vit_small_patch16_224.augreg_in21k', 'path': vit_s_path, 'dict_key': 'vit_s'},  # 
        # {'name': 'vit_base_patch16_224.augreg_in21k', 'path': vit_b_path, 'dict_key': 'vit_b'}, 
    ]

    num_experts = {}
    # num_experts['vit_t'] = n_EXPs
    num_experts['vit_s'] = n_EXPs
    # num_experts['vit_b'] = n_EXPs

    # Create models and store them in a dictionary
    vit_t = timm.create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=False, checkpoint_path=vit_t_path)
    # vit_s = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=False, checkpoint_path=vit_s_path)
    model_group = {}
    for config in model_configs:
        model = timm.create_model(config['name'], pretrained=False, checkpoint_path=config['path'])
        model_group[config['dict_key']] = model

    weight = {}
    layer_indices = range(12)
    for key in model_group:
        model = model_group[key]

        val_loader = prepare_imagenet(*dataset_arg, model)

        mlp_act = None
        attn_proj = None
        norm2 = None
        if not (select_type == 'random' or select_type == 'uniform'):
            print(f"Start feature analysis for {key}")
            print("analysis mlp.act")
            mlp_act = feature_analysis(model, layer_indices, val_loader, position='mlp.act')
            print("analysis attn.proj")
            attn_proj = feature_analysis(model, layer_indices, val_loader, position='attn.proj')
            print("analysis norm2")
            norm2 = feature_analysis(model, layer_indices, val_loader, position='norm2')
        selected_idx = generate_expert_indices(model, MoE_dict, num_experts[key], moe_blocks,
                                               mlp_act, attn_proj, norm2, select_type)

        is_glu = True if 'eva02' in key else False 
        weight[key]=get_selected_weight(
            model, 
            is_glu=is_glu,
            moe_blocks=moe_blocks,
            dense_channel_idx=selected_idx['dense_channel_idx'], 
            dense_attn_idx=selected_idx['dense_attn_idx'], 
            dense_mlp_idx=selected_idx['dense_mlp_idx'],
            moe_channel_idx=selected_idx['moe_channel_idx'], 
            moe_attn_idx=selected_idx['moe_attn_idx'], 
            moe_mlp_idx=selected_idx['moe_mlp_idx'],
            num_experts=num_experts[key], 
            select_unit=select_unit)
        # torch.save(weight[key], os.path.join(save_path, f'{key}-{select_type}_selected.pt'))

    Dense_part = vit_t.state_dict()
    # Dense_part = vit_s.state_dict()
    # Dense_part = weight['vit_s']
    if select_unit == 'block': 
        weight['eva02_mim'] = combine_eva_attn(weight['eva02_mim'], moe_blocks)

    if if_dualpath:
        weight['vit_s'] = convert_vit_to_dualpath(weight['vit_s'], moe_blocks, c_EXPs, u_EXPs, univ_factor, Dense_part=None)

    stitching = True
    MoE_ckpt = MoE_stitching(
        Dense_part, 
        MoE_eva=weight['eva02_mim'] if 'eva02_mim' in weight else None, 
        MoE_vit=weight['vit_s'], 
        MoE_dict=MoE_dict) if stitching else weight['vit_s']

    print('Finish weight selection. Start saving...')
    prefix = 'Dual-MoE' if if_dualpath else 'MoE'
    n_EXPs = n_EXPs if not if_dualpath else [c_EXPs, u_EXPs]
    torch.save(MoE_ckpt, os.path.join(save_path, f'{prefix}-{select_type}-{select_unit}-{n_EXPs}-{moe_blocks}_[D_vit_t]+[E_vit_s+univ_t]_selected.pt'))


# %%
if __name__=='__main__':
    main()


