nnictl create --config ./RetNet/configs/NNI/nni_config.yaml --port 6011
nnictl stop --all
nnictl experiment delete [7ay1q8hv]
nnictl view [7ay1q8hv] --port 6011
nnictl experiment list
nnictl experiment export [7ay1q8hv] -t csv -f work_dirs/nni/retnet-s-cifar10


NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh RetNet/configs/RetNet-S-p16_in1k.py 4 --work-dir ./work_dirs/RetNetS-woMask_in1k_b128x8_adamw1e-3

NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29511 ./tools/dist_train.sh softmoe/configs/timm/vit_tiny_timm.py 4 --work-dir ./work_dirs/ViT-Tiny-21kpretrained-256*4*4

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh softmoe/configs/softmoe/vit_tiny_baseline.py 4 --work-dir ./work_dirs/ViT-Tiny-baseline

python tools/analysis_tools/get_flops.py 
python ./tools/visualization/browse_dataset.py RetNet/configs/GANet-S-p16_in1k.py -m pipeline -n 100 -o ./vis_data