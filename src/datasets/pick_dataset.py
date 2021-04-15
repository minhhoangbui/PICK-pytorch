# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/9/2020 9:16 PM
import os
from typing import *
from pathlib import Path
import warnings
import random

import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import pandas as pd

from src.datasets import documents
from src.datasets.documents import Document
from src.utils.class_utils import ClassVocab, entities2iob_labels


sroie_entities_list = [
    "company",
    "address",
    "date",
    "total"
]

bizi_entities_list = [
    'buyer_address_line', 'issued_date', 'item_name',
    'total_amount_with_vat_in_words', 'invoice_number', 'buyer_legal_name',
    'vat_percentage', 'seller_address_line', 'template_code',
    'payment_method_name', 'total_price_before_vat', 'total_vat_amount',
    'total_amount_with_vat', 'invoice_series', 'total_amount_without_vat',
    'seller_legal_name', 'seller_tax_code', 'buyer_tax_code', 'unit',
    'unit_price', 'vat_amount', 'total_price_after_vat', 'discount_amount',
    'quantity', 'item_code', 'buyer_display_name'
]


class PICKDataset(data.Dataset):

    def __init__(self, dataset_name: str = None,
                 files_name: str = None,
                 boxes_and_transcripts_folder: str = 'boxes_and_transcripts',
                 images_folder: str = 'images',
                 entities_folder: str = 'entities',
                 iob_tagging_type: str = 'box_and_within_box_level',
                 resized_image_size: Tuple[int, int] = (480, 960),
                 ignore_error: bool = False,
                 training: bool = True
                 ):
        """
        :param dataset_name: Name of dataset
        :param files_name: containing training and validation samples list file.
        :param boxes_and_transcripts_folder: gt or ocr result containing transcripts, boxes and box entity type (optional).
        :param images_folder: whole images file folder
        :param entities_folder: exactly entity type and entity value of documents, containing json format file
        :param iob_tagging_type: 'box_level', 'document_level', 'box_and_within_box_level'
        :param resized_image_size: resize whole image size, (w, h)
        :param ignore_error:
        :param training: True for train and validation mode, False for test mode. True will also load labels,
        and files_name and entities_file must be set.
        """
        super().__init__()
        assert dataset_name is not None, "Have to specify dataset name"
        self.dataset_name = dataset_name
        self._image_ext = None
        self._ann_ext = None
        self.iob_tagging_type = iob_tagging_type
        self.ignore_error = ignore_error
        self.training = training
        assert resized_image_size and len(resized_image_size) == 2, 'resized image size not be set.'
        self.resized_image_size = tuple(resized_image_size)  # (w, h)

        self.entities_list = eval(f'{dataset_name}_entities_list')
        self.iob_labels_vocab_cls = ClassVocab(entities2iob_labels(self.entities_list),
                                               specials_first=False)

        if self.training:  # used for train and validation mode
            self.files_name = Path(files_name)
            self.data_root = self.files_name.parent
            self.boxes_and_transcripts_folder: Path = self.data_root.joinpath(boxes_and_transcripts_folder)
            self.images_folder: Path = self.data_root.joinpath(images_folder)
            self.entities_folder: Path = self.data_root.joinpath(entities_folder)
            if self.iob_tagging_type != 'box_level':
                if not self.entities_folder.exists():
                    raise FileNotFoundError('Entity folder is not exist!')
        else:  # used for test mode
            self.boxes_and_transcripts_folder: Path = Path(boxes_and_transcripts_folder)
            self.images_folder: Path = Path(images_folder)

        if not (self.boxes_and_transcripts_folder.exists() and self.images_folder.exists()):
            raise FileNotFoundError('Not contain boxes_and_transcripts folder {} or images folder {}.'
                                    .format(self.boxes_and_transcripts_folder.as_posix(),
                                            self.images_folder.as_posix()))
        if self.training:
            self.files_list = pd.read_csv(self.files_name.as_posix(), header=None,
                                          names=['index', 'file_name'],
                                          dtype={'index': int, 'file_name': str})
        else:
            self.files_list = list(self.boxes_and_transcripts_folder.glob('*.tsv'))

    def __len__(self):
        return len(self.files_list)

    def get_iob_labels_vocab(self):
        return self.iob_labels_vocab_cls

    def get_image_file(self, basename):
        """
        Return the complete name (fill the extension) from the basename.
        """
        if self._image_ext is None:
            filename = list(self.images_folder.glob(f'**/{basename}.*'))[0]
            self._image_ext = os.path.splitext(filename)[1]

        return self.images_folder.joinpath(basename + self._image_ext)

    def get_ann_file(self, basename):
        """
        Return the complete name (fill the extension) from the basename.
        """
        if self._ann_ext is None:
            filename = list(self.boxes_and_transcripts_folder.glob(f'**/{basename}.*'))[0]
            self._ann_ext = os.path.splitext(filename)[1]

        return self.boxes_and_transcripts_folder.joinpath(basename + self._ann_ext)

    def __getitem__(self, index):
        entities_file = None

        if self.training:
            data_item: pd.Series = self.files_list.iloc[index]
            # config file path
            boxes_and_transcripts_file = self.get_ann_file(data_item['file_name'])
            image_file = self.get_image_file(data_item['file_name'])

            entities_file = self.entities_folder.joinpath(data_item['file_name'] + '.txt')
        else:
            boxes_and_transcripts_file = self.get_ann_file(Path(self.files_list[index]).stem)
            image_file = self.get_image_file(Path(self.files_list[index]).stem)

        if not boxes_and_transcripts_file.exists() or not image_file.exists():
            if self.ignore_error and self.training:
                warnings.warn('{} is not exist. get a new one.'.format(boxes_and_transcripts_file))
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError('Sample: {} not exist.'.format(boxes_and_transcripts_file.stem))

        try:
            if self.training:
                document = documents.Document(boxes_and_transcripts_file, image_file, self.entities_list,
                                              self.resized_image_size, self.dataset_name, self.iob_tagging_type,
                                              entities_file, training=self.training)
            else:
                document = documents.Document(boxes_and_transcripts_file, image_file, self.entities_list,
                                              self.resized_image_size, self.dataset_name, image_index=index,
                                              training=self.training)
            return document
        except Exception as e:
            if self.ignore_error:
                warnings.warn('loading samples is occurring error, try to regenerate a new one.')
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError('Error occurs in image {}: {}'.format(boxes_and_transcripts_file.stem, e.args))


class BatchCollateFn:
    """
    padding input (List[Example]) with same shape, then convert it to batch input.
    """

    def __init__(self, training: bool = True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.training = training

    def __call__(self, batch_list: List[Document]):
        iob_tags_label_batch_tensor = None
        image_index_tensor = None

        # dynamic calculate max boxes number of batch,
        # this is suitable to one gpus or multi-nodes multi-gpus training mode,
        # due to pytorch distributed training strategy.
        max_boxes_num_batch = max([doc.boxes_num for doc in batch_list])
        max_transcript_len = max([doc.transcript_len for doc in batch_list])

        # fix MAX_BOXES_NUM and MAX_TRANSCRIPT_LEN. this ensures batch has same shape,
        # but lead to waste memory and slow speed..
        # this is suitable to one nodes multi gpus training mode, due to pytorch DataParallel training strategy
        # max_boxes_num_batch = documents.MAX_BOXES_NUM
        # max_transcript_len = documents.MAX_TRANSCRIPT_LEN

        # padding every sample with same shape, then construct batch_list samples

        # whole image, B, C, H, W
        image_batch_tensor = torch.stack([self.transform(x.whole_image) for x in batch_list], dim=0).float()

        # relation features, (B, num_boxes, num_boxes, 6)
        relation_features_padded_list = [F.pad(torch.FloatTensor(doc.relation_features),
                                               (0, 0, 0, max_boxes_num_batch - doc.boxes_num,
                                                0, max_boxes_num_batch - doc.boxes_num))
                                         for doc in batch_list]
        relation_features_batch_tensor = torch.stack(relation_features_padded_list, dim=0)

        # boxes coordinates,  (B, num_boxes, 8)
        boxes_coordinate_padded_list = [F.pad(torch.FloatTensor(doc.boxes_coordinate),
                                              (0, 0, 0, max_boxes_num_batch - doc.boxes_num))
                                        for doc in batch_list]
        boxes_coordinate_batch_tensor = torch.stack(boxes_coordinate_padded_list, dim=0)

        # text segments (B, num_boxes, transcript_len)
        text_segments_padded_list = [F.pad(torch.LongTensor(doc.text_segments[0]),
                                           (0, max_transcript_len - doc.transcript_len,
                                            0, max_boxes_num_batch - doc.boxes_num),
                                           value=doc.TextSegmentsField.vocab.stoi['<pad>'])

                                     for doc in batch_list]
        text_segments_batch_tensor = torch.stack(text_segments_padded_list, dim=0)

        # text length (B, num_boxes)
        text_length_padded_list = [F.pad(torch.LongTensor(doc.text_segments[1]),
                                         (0, max_boxes_num_batch - doc.boxes_num))
                                   for doc in batch_list]
        text_length_batch_tensor = torch.stack(text_length_padded_list, dim=0)

        # text mask, (B, num_boxes, transcript_len)
        mask_padded_list = [F.pad(torch.ByteTensor(doc.mask),
                                  (0, max_transcript_len - doc.transcript_len,
                                   0, max_boxes_num_batch - doc.boxes_num))
                            for doc in batch_list]
        mask_batch_tensor = torch.stack(mask_padded_list, dim=0)

        if self.training:
            # iob tag label for input text, (B, num_boxes, transcript_len)
            iob_tags_label_padded_list = [F.pad(torch.LongTensor(doc.iob_tags_label),
                                                (0, max_transcript_len - doc.transcript_len,
                                                 0, max_boxes_num_batch - doc.boxes_num),
                                                value=doc.IOBTagsField.vocab.stoi['<pad>'])
                                          for doc in batch_list]
            iob_tags_label_batch_tensor = torch.stack(iob_tags_label_padded_list, dim=0)

        else:
            # (B,)
            image_index_list = [x.image_index for x in batch_list]
            image_index_tensor = torch.tensor(image_index_list)

        # For easier debug.
        filenames = [doc.image_filename for doc in batch_list]

        # Convert the data into dict.
        if self.training:
            batch = dict(whole_image=image_batch_tensor,
                         relation_features=relation_features_batch_tensor,
                         text_segments=text_segments_batch_tensor,
                         text_length=text_length_batch_tensor,
                         boxes_coordinate=boxes_coordinate_batch_tensor,
                         mask=mask_batch_tensor,
                         iob_tags_label=iob_tags_label_batch_tensor,
                         filenames=filenames)
        else:
            batch = dict(whole_image=image_batch_tensor,
                         relation_features=relation_features_batch_tensor,
                         text_segments=text_segments_batch_tensor,
                         text_length=text_length_batch_tensor,
                         boxes_coordinate=boxes_coordinate_batch_tensor,
                         mask=mask_batch_tensor,
                         image_indexs=image_index_tensor,
                         filenames=filenames)

        return batch
