#!/bin/bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 \
--master_addr=127.0.0.1 --master_port=6705 \
scripts/train.py -c configs/config.json -d 3,4 --local_world_size 2
#--resume saved/models/PICK_Default/test_0401_124746/checkpoint-epoch100.pth

