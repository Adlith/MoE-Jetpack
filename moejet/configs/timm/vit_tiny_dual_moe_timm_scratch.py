_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['moejet'],
    allow_failed_imports=False)

# hyper_parameters
n_GPU = 4

total_batch = 4096
lr = 4e-3
bs = 256
accumulative_counts = total_batch//(n_GPU*bs)

dp = 0.1
moe_dr = 0.4
moe_dp = 0.1
moe_logits_drop = 0.0
slot_layernorm = True
c_EXPs = 98
u_EXPs = 196
univ_factor = 1/4
n_slots = 1

moe_groups_dict = {}
# phi_groups_dict = None
phi_groups_dict = {}
moe_layers = [6,7,8,9,10,11]

name = f'ViT-T-EXP_[{c_EXPs},{u_EXPs}]-slots_{n_slots}-dp_{dp}-moedr_{moe_dr}-moedpr_{moe_dp}-Nmoe_{moe_layers}-{total_batch}_{accumulative_counts}_{n_GPU}-lr_{lr}-softmax+entmax15'  # -noise-ld
moe_mult = 1.0

model = dict(
    type='TimmClassifier',
    model_name='vit_tiny_moe_custom',  # vit_tiny_patch16_224.augreg_in21k
    with_cp=False,
    num_classes=1000,
    pretrained=False,
    checkpoint_path='',
    global_pool='avg',  # token, avg
    # moe settings
    only_phi=False,
    moe_groups_dict=moe_groups_dict,
    phi_groups_dict=phi_groups_dict,
    moe_logits_drop=moe_logits_drop,
    slot_layernorm=slot_layernorm,
    moe_layers=moe_layers,
    slots_per_expert=n_slots,
    num_experts=197,
    # dualpath moe
    if_dualpath=True,
    core_experts=c_EXPs,
    univ_experts=u_EXPs,
    univ_factor=univ_factor,
    #
    layer_scale=False,
    moe_droprate=moe_dr,
    moe_drop_path_rate=moe_dp,
    drop_path_rate=dp,
    add_noise=True,
    noise_mult=1/(c_EXPs+u_EXPs),
    compress_ratio=1.0,
    loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)


# dataset settings
train_dataloader = dict(
    batch_size=bs,
    num_workers=8,
)

val_dataloader = dict(
    batch_size=bs,
    num_workers=8,
)

# schedule settings
DECAY_MULT = 0.0
def generate_moe_custom_keys(layer_keys, lr_mult=1.0, decay_mult=0.0):
    keys = {}
    lr_mult_param = ['mlp.phi', 'mlp.scales', 'mlp.slot_norm'
                    # 'mlp.core_experts.vit_fc1.weight', 'mlp.core_experts.vit_fc2.weight',
                    # 'mlp.univ_experts.vit_fc1.weight', 'mlp.univ_experts.vit_fc2.weight',
                    # 'norm2.weight', 'norm2.bias'
                     ]
    decay_mult_param = ['mlp.scales']

    for key in layer_keys:
        for param in lr_mult_param:
            full_key = f'blocks.{key}.{param}'
            if full_key not in keys:
                keys[full_key] = {}
            keys[full_key]['lr_mult'] = lr_mult
        
        for param in decay_mult_param:
            full_key = f'blocks.{key}.{param}'
            if full_key not in keys:
                keys[full_key] = {}
            keys[full_key]['decay_mult'] = decay_mult
    
    return keys

custom_keys = generate_moe_custom_keys(moe_layers, lr_mult=moe_mult, decay_mult=DECAY_MULT)
common_keys = {
    # '.cls_token': dict(decay_mult=DECAY_MULT),
    '.pos_embed': dict(decay_mult=DECAY_MULT),
}
custom_keys.update(common_keys)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    # constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        # layer_decay_rate=layer_decay_rate,
        norm_decay_mult=DECAY_MULT,
        bias_decay_mult=DECAY_MULT,
        custom_keys=custom_keys,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2),  # max_norm=0.1
    accumulative_counts=accumulative_counts,
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR', 
        T_max=280,
        eta_min_ratio=1e-2,
        by_epoch=True, 
        begin=20),
]

# runtime settings
# custom_hooks = [dict(type='EMAHook', momentum=1e-4)]
# custom_hooks = [
#     dict(type='Fp16CompresssionHook'),
# ]

train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=3)
val_cfg = dict()
test_cfg = dict()

randomness = dict(seed=3407, deterministic=False)
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=True,
)
default_hooks = dict(
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=1),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

work_dir = f'work_dirs/{name}'
# resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
# auto_scale_lr = dict(base_batch_size=4096)
