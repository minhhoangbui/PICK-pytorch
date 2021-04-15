# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/11/2020 10:02 PM

from typing import *
import numpy as np
import pandas as pd

from .span_based_f1 import SpanBasedF1Measure
import tabulate


class MetricTracker:
    def __init__(self, *keys, writer=None):
        """
        loss metric tracker
        :param keys:
        :param writer:
        """
        self.writer = writer
        columns = ['total', 'counts', 'average']
        self._data = pd.DataFrame(np.zeros((len(keys), len(columns))), index=keys, columns=columns)
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, epoch):
        if self.writer is not None:
            self.writer.add_scalar(key, value, epoch)
        self._data.total[key] += value
        self._data.counts[key] += 1
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class SpanBasedF1MetricTracker:
    """
    mEF metrics tracker
    """

    def __init__(self, vocab, **kwargs):
        self._metric = SpanBasedF1Measure(vocab=vocab, **kwargs)
        self.reset()

    def update(self, class_probabilities, tags, mask):
        self._metric(class_probabilities, tags, mask.float())

    def result(self):
        metric = self._metric.get_metric()
        data_dict = {}
        for k, v in metric.items():
            entity = k.split('-')[-1]

            item = data_dict.get(entity, {})
            if 'mEF' in k:
                item['mEF'] = v
            elif 'mEP' in k:
                item['mEP'] = v
            elif 'mER' in k:
                item['mER'] = v
            elif 'mEA' in k:
                item['mEA'] = v
            else:
                item['support'] = v
            data_dict[entity] = item

        return data_dict

    def reset(self):
        self._metric.reset()

    @staticmethod
    def dict2str(data_dict: Dict):
        data_list = [['name', 'mEP', 'mER', 'mEF', 'mEA', 'support']]
        for k, v in data_dict.items():
            data_list.append([k, v['mEP'], v['mER'], v['mEF'], v['mEA'], v['support']])
        return tabulate.tabulate(data_list, tablefmt='grid', headers='firstrow')
