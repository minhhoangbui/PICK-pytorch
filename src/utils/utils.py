# -*- coding: utf-8 -*-

from typing import *
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import torch

from .class_utils import keys_vocab_cls


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """ wrapper function for endless data loader. """
    for loader in repeat(data_loader):
        yield from loader


def iob_index_to_str(tags: List[List[int]], iob_labels_vocab_cls):
    decoded_tags_list = []
    for doc in tags:
        decoded_tags = []
        for tag in doc:
            s = iob_labels_vocab_cls.itos[tag]
            if s == '<unk>' or s == '<pad>':
                s = 'O'
            decoded_tags.append(s)
        decoded_tags_list.append(decoded_tags)
    return decoded_tags_list


def text_index_to_str(texts: torch.Tensor, mask: torch.Tensor):
    # union_texts: (B, num_boxes * T)
    union_texts = to_union(texts, mask, keys_vocab_cls)
    B, NT = union_texts.shape

    decoded_tags_list = []
    for i in range(B):
        decoded_text = []
        for text_index in union_texts[i]:
            text_str = keys_vocab_cls.itos[text_index]
            if text_str == '<unk>' or text_str == '<pad>':
                text_str = 'O'
            decoded_text.append(text_str)
        decoded_tags_list.append(decoded_text)
    return decoded_tags_list


def to_union(contents, mask, vocab_cls):
    """
    :param contents: (B, N, T)
    :param mask: (B, N, T)
    :param vocab_cls
    :return:
    """

    B, N, T = contents.shape

    contents = contents.reshape(B, N * T)
    mask = mask.reshape(B, N * T)

    # union tags as a whole sequence, (B, N*T)
    union_contents = torch.full_like(contents, vocab_cls['<pad>'], device=contents.device)

    max_seq_length = 0
    for i in range(B):
        valid_text = torch.masked_select(contents[i], mask[i].bool())
        valid_length = valid_text.size(0)
        union_contents[i, :valid_length] = valid_text

        if valid_length > max_seq_length:
            max_seq_length = valid_length

    # max_seq_length = documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN
    # (B, N*T)
    union_contents = union_contents[:, :max_seq_length]

    # (B, N*T)
    return union_contents

