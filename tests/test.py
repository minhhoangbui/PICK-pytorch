# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 10:53 AM

import argparse
import collections
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
import pandas as pd

from src.model import GLCN
from src.runner.config_parser import ConfigParser
import src.model.pick as pick_arch
from src.datasets import pick_dataset

from src.model.backbones import resnet
from src.datasets.pick_dataset import PICKDataset, BatchCollateFn


def test_glcn_model():
    batch_size, node_nums, in_dim, out_dim = 2, 5, 16, 32
    glcn = GLCN(in_dim, out_dim)
    x = torch.randn(batch_size, node_nums, in_dim)
    rel_features = torch.randn(batch_size, node_nums, node_nums, 6)
    adj = torch.randn(batch_size, node_nums, node_nums)
    box_num = torch.ones(batch_size, 1).int()
    x, soft_adj, gl_loss = glcn(x, rel_features, adj, box_num)
    print(x.shape, soft_adj.shape, gl_loss.shape)


def test_model():
    args = argparse.ArgumentParser(description='PICK parameters')
    args.add_argument('-c', '--config', default='../config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args, None)

    logger = config.get_logger('train')

    model = config.init_obj('model_arch', pick_arch)
    logger.info(model)


def test_resnet():
    model = resnet.resnet50()
    x = torch.randn(2, 3, 64, 128)
    x = model(x)
    print(x.shape)


def test_train_datasets():
    filename = Path(__file__).parent.parent.joinpath('/home/hoangbm/ner_data/sroie/test/test_list.csv').as_posix()
    train_dataset = PICKDataset(
        dataset_name='sroie',
        files_name=filename,
        iob_tagging_type='box_and_within_box_level',
        resized_image_size=(480, 960)
    )
    _ = train_dataset[4]

    data_loader = DataLoader(train_dataset, batch_size=10, collate_fn=BatchCollateFn(), num_workers=2)
    for idx, data_item in tqdm(enumerate(data_loader)):
        whole_image = data_item['whole_image']
        relation_features = data_item['relation_features']
        text_segments = data_item['text_segments']
        text_length = data_item['text_length']
        iob_tags_label = data_item['iob_tags_label']
        boxes_coordinate = data_item['boxes_coordinate']
        mask = data_item['mask']
        print('whole_image: ', whole_image.shape)
        print('relation_features: ', relation_features.shape)
        print('text_segments: ', text_segments.shape)
        print('text_length: ', text_length.shape)
        print('iob_tags_label: ', iob_tags_label.shape)
        print('boxes_coordinate: ', boxes_coordinate.shape)
        print('mask: ', mask.shape)
        exit()


def test_evaluate_datasets():
    bboxes_and_transcripts = '/home/hoangbm/ner_data/bizi/test/boxes_and_transcripts'
    images_dir = '/home/hoangbm/ner_data/bizi/test/images'
    test_dataset = PICKDataset(
        dataset_name='bizi',
        boxes_and_transcripts_folder=bboxes_and_transcripts,
        images_folder=images_dir,
        resized_image_size=(480, 960),
        ignore_error=False, training=False
    )
    _ = test_dataset[5]


def test_model_forward():
    # torch.backends.cudnn.benchmark = False
    args = argparse.ArgumentParser(description='PICK parameters')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type target help')
    options = [
        CustomArgs(['--local_world_size'], default=1, type=int, target='local_world_size',
                   help='this is passed in explicitly'),
        CustomArgs(['--local_rank'], default=0, type=int, target='local_rank',
                   help='this is automatically passed in via launch.py')

    ]
    config = ConfigParser.from_args(args, options)

    if torch.cuda.is_available() and config['local_rank'] != -1:
        torch.cuda.set_device(config['local_rank'])
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    pick_model = config.init_obj('model_arch', pick_arch)
    pick_model.to(device)

    dataset = config.init_obj('train_dataset', pick_dataset)

    # filename = Path(__file__).parent.parent.joinpath('data/data_examples_root/train_samples_list.csv').as_posix()
    # dataset = PICKDataset(files_name=filename,
    #                       iob_tagging_type = 'box_level',
    #                       resized_image_size = (480, 960))

    data_loader = DataLoader(dataset, batch_size=1, collate_fn=BatchCollateFn(), num_workers=2)
    for idx, data_item in tqdm(enumerate(data_loader)):
        for key, tensor in data_item.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                data_item[key] = tensor.to(device, non_blocking=True)
        output = pick_model(**data_item)

        logits = output['logits']
        new_mask = output['new_mask']
        # gl_loss = output['gl_loss']
        # crf_loss = output['crf_loss']
        predictions = []
        if hasattr(pick_model, 'module'):
            best_paths = pick_model.module.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                          logits_batch_first=True)
        else:
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                   logits_batch_first=True)
        print(best_paths)
        for path, score in best_paths:
            predictions.append(path)
        print(predictions)
        exit()
        # print(logits.shape)


def test_read_csv():
    filename = r'/home/Wen/data/code/PICK/PICK-pytorch/data/data_examples_root/train_samples_list.csv'
    # filename = r'/home/Wen/data/code/PICK/PICK-pytorch/data/data_examples_root/baseline_test.csv'
    res = pd.read_csv(filename, header=None,
                      names=['index', 'document_class', 'file_name'])
    item = res.iloc[0]
    print(item)


def test_metrics():
    import numpy as np
    keys = ('loss', 'gl_loss', 'crf_loss')
    columns = ['total', 'counts', 'average']
    data = pd.DataFrame(np.zeros((len(keys), len(columns))), index=keys, columns=columns)
    print(data.index)


# used for data-parallel mode training, distributed-data-parallel model can change it for speed up.
# 1. pick_dataset.py BatchCollateFn __call__ line 190  max_boxes_num_batch & max_transcript_len
# 2. graph.py GraphLearningLayer line 51 mask = self.compute_static_mask(box_num) instead of compute_dynamic_mask func
# 3. decoder.py UnionLayer line 142 max_doc_seq_len = doc_seq_len.max()
# 4. decoder.py BiLSTMLayer pad_packed_sequence line 105, total_length set to max_length
# 5. utils.py iob_tags_to_union_iob_tags, texts_to_union_texts, comment max_seq_length =
#    documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN

if __name__ == '__main__':
    # test_glcn_model()
    # test_model()
    # test_resnet()
    # test_train_datasets()
    test_evaluate_datasets()
    # test_model_forward()
    # test_metrics()
