#!/bin/bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 \
--master_addr=127.0.0.1 --master_port=29506 \
scripts/train.py -c configs/config.json -d 4,5 --local_world_size 2
#--resume /tmp/PICK/models/shufflenet_bizi/model_best.pth