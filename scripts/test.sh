#!/bin/bash
python test.py --checkpoint saved/models/PICK_Default/test_0401_124746/checkpoint-epoch100.pth \
               --boxes_transcripts /home/hoangbm/PICK-pytorch/data/test_data_example/boxes_and_transcripts \
               --images_path /home/hoangbm/PICK-pytorch/data/test_data_example/images \
               --output_folder /tmp/PICK \
               --gpu 0 --batch_size 2