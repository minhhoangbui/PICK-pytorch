#!/bin/bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=3 \
--master_addr=127.0.0.1 --master_port=29505 \
scripts/train.py -c configs/config.json -d 0,1,2 --local_world_size 3
#--resume saved/models/PICK_Default/test_0401_124746/checkpoint-epoch100.pth