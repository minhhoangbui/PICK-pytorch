#!/bin/bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 \
--master_addr=127.0.0.1 --master_port=29505 \
 train.py -c config.json -d 5,1,3,4 --local_world_size 4
#--resume saved/models/PICK_Default/test_0401_124746/checkpoint-epoch100.pth