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
import copy
import pickle
import argparse

torch.random.manual_seed(3407)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE_ID = [0,1,2,3,4,5,6,7]


def prepare_imagenet(imagenet_root, val_batch_size, num_workers, model, sample_size=128*300):
    from timm.data import resolve_model_data_config, create_transform

    data_config = resolve_model_data_config(model)
    dataset = ImageFolder(os.path.join(imagenet_root, 'train'),
                          transform=create_transform(**data_config, is_training=False))
    
    # 可配置的样本选择方式，允许用户传入sample_size参数
    indices = torch.randperm(len(dataset))[:sample_size]
    subset = Subset(dataset, indices)

    return DataLoader(subset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

def batch_mean(data, batch_size=100):
    data_batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    means = [np.mean(batch, axis=0) for batch in data_batches]
    return np.mean(means, axis=0)


def feature_analysis(model, val_loader, position='mlp.act', threshold_method='percentile', percentile=80):
    
    layer_indices = range(len(model.blocks))
    model = model.to(DEVICE)
    model = nn.DataParallel(model, device_ids=DEVICE_ID)
    model.eval()

    # 初始化存储结构
    activation_stats = {
        idx: {
            'L1': 0.0,
            'L2': 0.0,
            'act_patrn': 0.0,
            'co_act': 0.0,
        } for idx in layer_indices
    }

    # 定义钩子函数
    def hook_function(idx, module, input, output):
        data = input[0] if use_input else output
        data = data.detach().cpu()

        data_cpu = data.view(-1).numpy()

        if threshold_method == 'mean_std':
            # 使用均值加标准差作为阈值
            mean = data_cpu.mean()
            std = data_cpu.std()
            threshold = mean + std  # 84.13% normal distribution
        elif threshold_method == 'percentile':
            # 使用百分位数作为阈值
            threshold = np.percentile(data_cpu, percentile)
        elif threshold_method == 'fixed':
            # 使用固定阈值
            threshold = 0.0
        else:
            raise ValueError('Invalid threshold_method')

        activation_map = (data > threshold).float()  # Shape: (b, n, d)

        co_act = einsum(activation_map, activation_map, 'b n d1, b n d2-> d1 d2')  # Shape: (b, d, d)
        co_act.fill_diagonal_(0)

        l1_norm = data.norm(p=1, dim=(0, 1))
        l2_norm = data.norm(p=2, dim=(0, 1))
        activation_pattern = activation_map.sum(dim=(0, 1))

        stats = activation_stats[idx]

        n_batch_samples = data.shape[0] * data.shape[1]
        stats['L1'] += l1_norm / n_batch_samples
        stats['L2'] += l2_norm / n_batch_samples
        stats['act_patrn'] += activation_pattern / n_batch_samples
        stats['co_act'] += co_act / n_batch_samples

    # 决定是否使用输入
    use_input = position == 'attn.proj'

    # 注册钩子
    hooks = []
    for idx in layer_indices:
        if position == 'mlp.act':
            target_layer = model.module.blocks[idx].mlp.act
        elif position == 'mlp.fc2':
            target_layer = model.module.blocks[idx].mlp.fc2
        elif position == 'attn.proj':
            target_layer = model.module.blocks[idx].attn.proj
        elif position == 'norm2':
            target_layer = model.module.blocks[idx].norm2
        else:
            raise ValueError('Invalid position argument')
        hook = target_layer.register_forward_hook(functools.partial(hook_function, idx))
        hooks.append(hook)

    # 运行推理
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Processing validation images"):
            images = images.to(DEVICE)
            model(images)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    return activation_stats


def generate_mlp_indices(
    num_layers,
    moe_layers,
    num_experts,
    total_size,
    subset_ratio,
    device=None,
    method='random',
    analysis_feature=None,
    metric_type='L1',
    select_global=False,
    method_kwargs=None
):
    """
    为每个 MLP 层生成索引字典，可以指定不同的索引生成方法。

    参数:
    - num_layers:模型的总层数。
    - num_experts:专家的数量。
    - moe_layers:使用 MoE 的层的索引列表。
    - total_size:MLP 层的总大小（例如，MLP 层的隐藏维度）。
    - subset_ratio:子集比例，用于计算 subset_size = int(total_size * subset_ratio)。
    - device:设备，可选。
    - method:索引生成方法，字符串，'random'、'uniform'、'importance'、'graph'。
    - analysis_feature:包含每层统计信息的字典，当 method='importance' 或 'graph' 时需要提供。
    - metric_type:使用的统计指标类型，'L1'、'L2'、'act_patrn' 或 'co_act'，当 method='importance' 或 'graph' 时需要。
    - select_global:是否全局选择单元，当 method='importance' 或 'graph' 时使用。
    - method_kwargs:索引生成方法的附加参数，字典类型。

    返回:
    - mlp_indices_dict:包含每个 MLP 层索引的字典。
    """
    import torch
    import numpy as np

    if not (0 < subset_ratio <= 1):
        raise ValueError("subset_ratio 必须在 (0, 1] 范围内。")
    subset_size = max(1, int(total_size * subset_ratio))

    method_kwargs = method_kwargs or {}
    mlp_indices_dict = {}

    # 准备必要的数据
    if method in ['importance', 'graph']:
        assert analysis_feature is not None, f"当 method='{method}' 时，需要提供 analysis_feature"

        if select_global:
            method_kwargs['select_global'] = True
            # 全局处理
            if method == 'importance':
                importance_scores_list = []
                for layer_idx in range(num_layers):
                    layer_stats = analysis_feature.get(layer_idx)
                    if layer_stats is None:
                        raise ValueError(f"在 analysis_feature 中未找到第 {layer_idx} 层的统计信息。")
                    importance_scores = layer_stats[metric_type]
                    importance_scores = torch.tensor(importance_scores, device=device)
                    # 对每一层的重要性分数进行归一化
                    epsilon = 1e-8
                    importance_scores = importance_scores / (importance_scores.sum() + epsilon)
                    importance_scores_list.append(importance_scores)
                # 对所有层的归一化重要性分数进行平均
                global_importance_scores = torch.stack(importance_scores_list).mean(dim=0)
                method_kwargs['importance_scores'] = global_importance_scores.to(device)

            elif method == 'graph':
                # 计算全局的 co_act_matrix
                co_act_matrices = []
                for layer_idx in range(num_layers):
                    layer_stats = analysis_feature.get(layer_idx)
                    if layer_stats is None:
                        raise ValueError(f"在 analysis_feature 中未找到第 {layer_idx} 层的统计信息。")
                    co_act_matrix = layer_stats[metric_type]
                    co_act_matrix = np.array(co_act_matrix)
                    np.fill_diagonal(co_act_matrix, 0)
                    co_act_matrices.append(co_act_matrix)
                total_co_act = np.sum(co_act_matrices, axis=0)
                method_kwargs['co_act_matrix'] = total_co_act

        else:
            # 层内处理，准备每层的数据
            layer_data_dict = {}
            for layer_idx in range(num_layers):
                layer_stats = analysis_feature.get(layer_idx)
                if layer_stats is None:
                    raise ValueError(f"在 analysis_feature 中未找到第 {layer_idx} 层的统计信息。")
                if method == 'importance':
                    importance_scores = layer_stats[metric_type]
                    importance_scores = torch.tensor(importance_scores, device=device)
                    layer_data_dict[layer_idx] = importance_scores
                elif method == 'graph':
                    co_act_matrix = layer_stats[metric_type]
                    co_act_matrix = co_act_matrix.numpy()
                    layer_data_dict[layer_idx] = co_act_matrix

    for layer_idx in range(num_layers):
        current_method_kwargs = method_kwargs.copy()
        if method in ['importance', 'graph'] and not select_global:
            # 添加层内的数据
            if method == 'importance':
                current_method_kwargs['importance_scores'] = layer_data_dict[layer_idx]
            elif method == 'graph':
                co_act_matrix = layer_data_dict[layer_idx]
                current_method_kwargs['co_act_matrix'] = co_act_matrix
                graph_hyp = current_method_kwargs.get('graph_hyp', 1.1)  # ['louvain', 1.05-1.1] ['metis', 12]
                graph_method = current_method_kwargs.get('graph_method', 'louvain')
                # Compute probabilities once per layer
                probabilities, part_to_nodes, subgraph_ids = generate_graph_prob(co_act_matrix, graph_method, graph_hyp)
                current_method_kwargs['probabilities'] = probabilities
                current_method_kwargs['part_to_nodes'] = part_to_nodes
                current_method_kwargs['subgraph_ids'] = subgraph_ids

        if layer_idx in moe_layers and num_experts > 0:
            # 为每个专家生成独立的索引
            per_expert_indices = {}
            for expert_idx in tqdm(range(num_experts), desc="generate neuron indexs"):
                indices = generate_indices(
                    total_size,
                    subset_size,
                    device=device,
                    method=method,
                    **current_method_kwargs
                )
                per_expert_indices[expert_idx] = indices
            mlp_indices_dict[layer_idx] = per_expert_indices
        else:
            pass
            # 为非 MoE 层或单专家的 MoE 层生成索引
            # indices = generate_indices(
            #     total_size,
            #     subset_size,
            #     device=device,
            #     method=method,
            #     **current_method_kwargs
            # )
            # mlp_indices_dict[layer_idx] = indices

    return mlp_indices_dict


def generate_indices(total_size, subset_size, device=None, method='random', **kwargs):
    """
    通用的索引生成函数，根据指定的方法生成索引。

    参数:
    - total_size:总的单位数量（如 MLP 层的尺寸）。
    - subset_size:子集的大小。
    - device:设备，可选。
    - method:索引生成方法，字符串，'random'、'uniform'、'importance'、'graph'。
    - **kwargs:其他传递给索引生成方法的参数。

    返回:
    - indices:生成的索引张量。
    """
    if method == 'random':
        indices = torch.randperm(total_size, device=device)[:subset_size]
    elif method == 'uniform':
        indices = torch.linspace(0, total_size - 1, steps=subset_size, device=device).long()
    elif method == 'importance':
        indices = importance_based_selection(total_size, subset_size, device=device, **kwargs)
    elif method == 'graph':
        indices = graph_based_selection(total_size, subset_size, device=device, **kwargs)
    else:
        raise ValueError(f"未知的索引生成方法:{method}")

    return torch.sort(indices.unique()).values
    # return indices



def importance_based_selection(total_size, subset_size, device=None, **kwargs):

    importance_scores = kwargs.get('importance_scores')
    select_global = kwargs.get('select_global', False)

    importance_scores = importance_scores.to(device)

    if select_global:
        _, indices = torch.topk(importance_scores, subset_size, largest=True)

    else:
        indices = torch.multinomial(importance_scores, subset_size, replacement=False)

    return indices


def graph_based_selection(total_size, subset_size, device=None, **kwargs):
    probabilities = kwargs.get('probabilities', None)
    part_to_nodes = kwargs.get('part_to_nodes', None)
    subgraph_ids = kwargs.get('subgraph_ids', None)
    select_global = kwargs.get('select_global', False)
    subgraph_number = min(len(probabilities), int(subset_size / total_size * len(probabilities) + 1)*3)

    if select_global:
        _, sampled_indices = torch.topk(probabilities, subset_size, largest=True)
    else:
        sampled_indices = torch.multinomial(probabilities, subgraph_number, replacement=False)

    # Select nodes
    selected_nodes = [node for i in sampled_indices for node in part_to_nodes[subgraph_ids[i]]]
    if len(selected_nodes) < subset_size:
        raise ValueError(f"selected_nodes:{len(selected_nodes)} < subset_size:{subset_size}")
    indices = torch.tensor(selected_nodes[:subset_size], device=device)

    return indices


def generate_graph_prob(co_act_matrix: np.ndarray, graph_method, graph_hyp):
    graph = nx.from_numpy_array(co_act_matrix)
    graph.remove_edges_from(nx.selfloop_edges(graph))  # 移除自环

    # # 验证边权重是否正确
    # for u, v, data in graph.edges(data=True):
    #     print(f"Edge ({u}, {v}) 的权重是 {data['weight']}")
    #     break  # 只打印第一条边的信息

    _, part_to_nodes = apply_graph_partitioning(graph, graph_method, graph_hyp)
    
    coactivation_sums = {}
    for part, nodes in part_to_nodes.items():
        if len(nodes) < 2:
            coactivation_sums[part] = 0.0
            continue
        submatrix = co_act_matrix[np.ix_(nodes, nodes)]
        # 去除对角线元素
        coactivation_sum = submatrix.sum() - np.trace(submatrix)
        coactivation_sums[part] = coactivation_sum

    subgraph_ids, coactivations = zip(*coactivation_sums.items())
    total = sum(coactivations)
    if total == 0:
        probabilities = torch.zeros(len(coactivations), dtype=torch.float)
    else:
        probabilities = torch.tensor(coactivations, dtype=torch.float) / total
    return probabilities, part_to_nodes, subgraph_ids


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

def apply_graph_partitioning(graph, method, hyper):
    if method == 'spectral':
        # 光谱聚类方法参数
        num_clusters = hyper
        adjacency_matrix = nx.adjacency_matrix(graph).toarray()
        sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', n_init=100)
        labels = sc.fit_predict(adjacency_matrix)
        partition = {node: label for node, label in enumerate(labels)}
    
    elif method == 'metis':
        num_partitions = int(hyper)
        edges = [(u, v, data.get('weight', 1)) for u, v, data in graph.edges(data=True)]
        num_nodes = graph.number_of_nodes()
        # num_connected_components = nx.number_connected_components(graph)

        # # 检查分区数量
        # max_possible_partitions = min(num_nodes, num_connected_components)
        # if num_partitions > max_possible_partitions:
        #     print(f"警告: 分区数量 {num_partitions} 超过了最大可能的分区数量 {max_possible_partitions}，将其设置为 {max_possible_partitions}")
        #     num_partitions = max_possible_partitions

        xadj = [0]
        adjncy = []
        eweights = []
        adj_dict = {i: [] for i in range(num_nodes)}
        for u, v, weight in edges:
            adj_dict[u].append((v, weight))
            adj_dict[v].append((u, weight))
        for i in range(num_nodes):
            neighbors = adj_dict[i]
            adjncy.extend([neighbor for neighbor, _ in neighbors])
            eweights.extend([int(weight) for _, weight in neighbors])
            xadj.append(len(adjncy))
        _, parts = pymetis.part_graph(num_partitions, xadj=xadj, adjncy=adjncy, eweights=eweights)
        partition = {node: part for node, part in enumerate(parts)}

    elif method == 'louvain':
        # Louvain 方法参数
        resolution = hyper
        partition = community_louvain.best_partition(graph, resolution=resolution, random_state=42)
        size = len(set(partition.values()))
        print(f"Resolution {resolution}: {size} communities detected")
    
    else:
        raise ValueError("Unsupported partitioning method")

    # 计算评价指标
    total_internal_weight = 0
    total_cut_weight = 0
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        if partition[u] == partition[v]:
            total_internal_weight += weight
        else:
            total_cut_weight += weight

    total_edge_weight = total_internal_weight + total_cut_weight
    edge_cut_ratio = total_cut_weight / total_edge_weight if total_edge_weight > 0 else 0

    # 计算模块度
    modularity = community_louvain.modularity(partition, graph, weight='weight')
    
    # print(f"总内部边权重: {total_internal_weight}")
    # print(f"总跨分区边权重: {total_cut_weight}")
    print(f"边割比率（Edge Cut Ratio）: {edge_cut_ratio:.4f}")
    print(f"模块度（Modularity）: {modularity:.4f}")
    print(f"  ")
    
    part_to_nodes = {}
    for node, part in partition.items():
        if part not in part_to_nodes:
            part_to_nodes[part] = []
        part_to_nodes[part].append(node)
    
    return partition, part_to_nodes


def get_ratio_str(subset_ratio):
    return f'{subset_ratio:.2f}'.replace('.', 'p')


def process_key_mlp(key, subset_ratio, prefix='experts.'):
    ratio_str = get_ratio_str(subset_ratio)
    key = re.sub(r'(blocks\.\d+\.mlp\.)', fr'\1{prefix}{ratio_str}.', key)
    return key


def select_and_stack_weights(dense_model, selected_indices, subset_experts_mapping, moe_layers, device=None):
    """
    从 dense 模型中根据 selected_indices 选择权重，并堆叠成 MoE 专家的权重。

    参数:
    - dense_model: 预训练的 dense 模型。
    - selected_indices: generate_mlp_indices 函数生成的索引字典。
    - subset_experts_mapping: subset_ratio 到 num_experts 的映射。
    - moe_layers: 使用 MoE 的层的索引列表。
    - device: 设备，可选。

    返回:
    - MoE_dict: 包含专家权重的字典，用于单独保存。
    """
    MoE_dict = {}
    state_dict = dense_model.cpu().state_dict()

    for subset_ratio, num_experts in subset_experts_mapping.items():
        for layer_idx in moe_layers:
            # 预先获取所有相关权重
            fc1_weight_key = f'blocks.{layer_idx}.mlp.fc1.weight'
            fc2_weight_key = f'blocks.{layer_idx}.mlp.fc2.weight'
            fc1_bias_key = f'blocks.{layer_idx}.mlp.fc1.bias'
            fc2_bias_key = f'blocks.{layer_idx}.mlp.fc2.bias'

            fc1_weight = state_dict[fc1_weight_key]
            fc2_weight = state_dict[fc2_weight_key]
            fc1_bias = state_dict.get(fc1_bias_key, None)
            fc2_bias = state_dict.get(fc2_bias_key, None)  # 通常不需要选择

            # 初始化专家权重列表
            fc1_experts = []
            fc2_experts = []
            fc1_bias_experts = [] if fc1_bias is not None else None
            fc2_bias_experts = [] if fc2_bias is not None else None

            # 遍历每个专家
            for expert_idx in tqdm(range(num_experts), desc=f"Processing subset_ratio {subset_ratio:.2f} experts in layer {layer_idx}"):
                indices = selected_indices[subset_ratio][layer_idx][expert_idx]

                # 选择并复制权重
                selected_fc1_weight = torch.index_select(fc1_weight, dim=0, index=indices).clone().to(device)
                selected_fc2_weight = torch.index_select(fc2_weight, dim=1, index=indices).clone().to(device)
                selected_fc1_bias = torch.index_select(fc1_bias, dim=0, index=indices).clone().to(device)

                fc1_experts.append(selected_fc1_weight)
                fc2_experts.append(selected_fc2_weight)
                fc1_bias_experts.append(selected_fc1_bias)
                
                fc2_bias_experts.append(fc2_bias)

            fc1_experts_tensor = torch.stack(fc1_experts, dim=0)  # (num_experts, input_size, subset_size)
            fc2_experts_tensor = torch.stack(fc2_experts, dim=0)  # (num_experts, subset_size, input_size)
            fc1_bias_experts_tensor = torch.stack(fc1_bias_experts, dim=0)  # (num_experts, subset_size)
            fc2_bias_experts_tensor = torch.stack(fc2_bias_experts, dim=0)  # (num_experts, subset_size)

            moe_fc1_weight_key = process_key_mlp(fc1_weight_key, subset_ratio)
            moe_fc2_weight_key = process_key_mlp(fc2_weight_key, subset_ratio)
            moe_fc1_bias_key = process_key_mlp(fc1_bias_key, subset_ratio)
            moe_fc2_bias_key = process_key_mlp(fc2_bias_key, subset_ratio)

            MoE_dict[moe_fc1_weight_key] = fc1_experts_tensor
            MoE_dict[moe_fc2_weight_key] = fc2_experts_tensor
            MoE_dict[moe_fc1_bias_key] = fc1_bias_experts_tensor
            MoE_dict[moe_fc2_bias_key] = fc2_bias_experts_tensor

    return MoE_dict


def remove_moe_weights(state_dict, moe_layers):
    """
    从 state_dict 中移除与 MoE 层相关的权重键。

    参数:
    - state_dict: 模型的 state_dict。
    - moe_layers: 使用 MoE 的层的索引列表。
    """
    keys_to_remove = []
    for layer_idx in moe_layers:
        keys_to_remove.extend([
            f'blocks.{layer_idx}.mlp.fc1.weight',
            f'blocks.{layer_idx}.mlp.fc2.weight',
            f'blocks.{layer_idx}.mlp.fc1.bias',
            f'blocks.{layer_idx}.mlp.fc2.bias'
        ])
    for key in keys_to_remove:
        state_dict.pop(key, None)  # 使用 pop 避免 KeyError


def get_all_key_paths(d, parent_key=''):
    items = []
    for key, value in d.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(get_all_key_paths(value, full_key))
        else:
            items.append((full_key, value))
    return items


def save_analysis_feature(analysis_feature, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(analysis_feature, f)

def load_analysis_feature(save_path):
    with open(save_path, 'rb') as f:
        return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="MoE Model Weight Initialization Script")

    # Paths
    parser.add_argument('--data_path', type=Path, default=Path('data/imagenet'),
                        help="Path to the ImageNet data directory")
    parser.add_argument('--save_path', type=Path, default=Path('moejet/weights/gen_weight'),
                        help="Path to save the generated MoE weights")
    parser.add_argument('--moe_config_path', type=Path, default=Path('moejet/configs/timm/vit_t_dynamicMoE_timm_21k_ft.py'),
                        help="Path to the MoE configuration file")
    parser.add_argument('--ckpt_name', type=str, default='vit_tiny_patch16_224.augreg_in21k',
                        help="Checkpoint name for the dense model")

    # Data loader parameters
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help="Batch size for validation")
    parser.add_argument('--sample_multiplier', type=int, default=300,
                        help="Multiplier to compute sample size (sample_size = val_batch_size * sample_multiplier)")

    # Analysis parameters
    parser.add_argument('--position', type=str, default='mlp.act',
                        help="Position for feature analysis")
    parser.add_argument('--threshold_method', type=str, default='percentile',
                        choices=['percentile', 'mean_std', 'fixed'],
                        help="Threshold method for feature analysis")
    parser.add_argument('--percentile', type=float, default=90.0,
                        help="Percentile value for thresholding")

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use for computations (e.g., cpu, cuda)")

    args = parser.parse_args()
    return args


def compare_state_dicts_keys(state_dict1, state_dict2):
    # 提取 keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    # 找出相同和不同的 keys
    common_keys = keys1.intersection(keys2)
    only_in_state_dict1 = keys1 - keys2
    only_in_state_dict2 = keys2 - keys1
    
    # 过滤出相同 keys 中带有特定关键词的部分
    keywords = ['phi', 'scale', 'scales', 'slot', 'experts']
    filtered_common_keys = [key for key in common_keys if any(kw in key for kw in keywords)]
    
    # 打印结果
    print("与MoE相同的 keys:")
    for key in filtered_common_keys:
        print(f"  - {key}")
    
    print("\n仅在 state_dict1 中的 keys:")
    for key in only_in_state_dict1:
        print(f"  - {key}")
    
    print("\n仅在 state_dict2 中的 keys:")
    for key in only_in_state_dict2:
        print(f"  - {key}")


def main(args):
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent

    path = current_dir.parents[1]
    sys.path.append(str(path))

    data_path = path / args.data_path
    val_batch_size = args.val_batch_size
    sample_size = val_batch_size * args.sample_multiplier

    save_path = path / args.save_path
    moe_config_path = path / args.moe_config_path

    ckpt_name = args.ckpt_name
    ckpt_path = path / 'data' / 'hgface' / f'{ckpt_name}.pth'
    dense_model = timm.create_model(ckpt_name, pretrained=False, checkpoint_path=ckpt_path, num_classes=1000)

    save_path.mkdir(parents=True, exist_ok=True)
    analysis_feature_filename = f'analysis_feature_batch{val_batch_size}_sample{sample_size}.pkl'
    analysis_feature_path = save_path / analysis_feature_filename

    moe_cfg = Config.fromfile(moe_config_path)
    MoE = build_classifier(moe_cfg['model'])
    MoE_dict = MoE.state_dict()

    moe_layers = moe_cfg.model.moe_layers
    subset_experts_mapping = moe_cfg.subset_experts_mapping

    recycle_method = moe_cfg.recycle_method
    metric_type = moe_cfg.metric_type

    if analysis_feature_path.exists():
        print(f"加载已存储的 analysis_feature: {analysis_feature_path}")
        analysis_feature = load_analysis_feature(analysis_feature_path)
    else:
        print("未找到已存储的 analysis_feature，开始新计算...")
        dataset_arg = (data_path, val_batch_size, 8)
        val_loader = prepare_imagenet(*dataset_arg, dense_model, sample_size=sample_size)

        analysis_feature = feature_analysis(
            dense_model,
            val_loader,
            position=args.position,
            threshold_method=args.threshold_method,
            percentile=args.percentile
        )
        save_analysis_feature(analysis_feature, analysis_feature_path)
        print(f"analysis_feature 已保存至 {analysis_feature_path}")

    num_layers = len(dense_model.blocks)
    total_size = dense_model.blocks[0].mlp.fc1.out_features

    selected_indices = {}
    for subset_ratio, num_experts in subset_experts_mapping.items():
        selected_indices[subset_ratio] = generate_mlp_indices(
            num_layers=num_layers,
            moe_layers=moe_layers,
            num_experts=num_experts,
            total_size=total_size,
            subset_ratio=float(subset_ratio),
            method=recycle_method,
            analysis_feature=analysis_feature,
            metric_type=metric_type
        )

    MoE_expert_weights = select_and_stack_weights(
        dense_model=dense_model,
        selected_indices=selected_indices,
        subset_experts_mapping=subset_experts_mapping,
        moe_layers=moe_layers,
        device='cpu'
    )

    # 加载 dense 模型的 state_dict
    recycle_state_dict = copy.deepcopy(dense_model.state_dict())

    # 移除与 MoE 层相关的权重
    remove_moe_weights(recycle_state_dict, moe_layers)
    recycle_state_dict.update(MoE_expert_weights)

    compare_state_dicts_keys(MoE_dict, recycle_state_dict)

    # 保存初始化后的 MoE 模型权重
    output_filename = f'DynamicMoE_T_{recycle_method}_{metric_type}_Exps{subset_experts_mapping}-MoElayer{moe_layers}.pth'
    torch.save(recycle_state_dict, save_path / output_filename)
    print(f"MoE 模型权重已保存至 {save_path / output_filename}")

if __name__ == "__main__":
    args = parse_args()
    main(args)