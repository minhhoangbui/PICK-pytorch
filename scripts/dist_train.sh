#!/bin/bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
--master_addr=127.0.0.1 --master_port=29505 \
scripts/train.py -c configs/config.json -d 1 --local_world_size 1
#--resume /tmp/PICK/models/shufflenet_sroie/model_best.pth