# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 9:26 PM


from collections import Counter
from torchtext.vocab import Vocab
from pathlib import Path


class ClassVocab(Vocab):

    def __init__(self, classes, **kwargs):
        """
        convert key to index(stoi), and get key string by index(itos)
        :param classes: list or str, key string or entity list
        :param kwargs:
        """
        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)
        elif isinstance(classes, list):
            cls_list = classes
        c = Counter(cls_list)
        specials = ['<pad>', '<unk>']
        self.special_count = len(specials)
        super().__init__(c, specials=specials, **kwargs)


def entities2iob_labels(entities: list):
    """
    get all iob string label by entities
    :param entities:
    :return:
    """
    tags = []
    for e in entities:
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
    tags.append('O')
    return tags


keys_vocab_cls = ClassVocab(Path(__file__).parent.joinpath('keys.txt'), specials_first=False)
