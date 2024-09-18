from PIL import Image
import torch as tr
import random
import torch
from torchvision import datasets, transforms


class SimpleDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        # super().__init__(vars(dataset))
        self.paths = dataset.paths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.paths)

    @property
    def len(self):
        return len(self.paths)

    def _create_iter_index(self):
        self._select_index = list(range(self.len))
        random.shuffle(self._select_index) if self.shuffle else None
        self._batches_num = self.len // self.batch_size + (self.len % self.batch_size != 0)
        self._select_index_batches = []
        for index in range(self._batches_num):
            start = self.batch_size * index
            end = self.batch_size * (index + 1) if self.batch_size * (index + 1) <= self.len else None
            self._select_index_batches.append(self._select_index[start:end])

    def __iter__(self):
        self._create_iter_index()
        self._current_batch_index = 0
        return self

    def __next__(self):
        if self._current_batch_index >= self._batches_num:
            raise StopIteration
        paths_output_list = []
        for index in self._select_index_batches[self._current_batch_index]:
            paths_output_list.append(self.paths[index])
        self._current_batch_index += 1
        return paths_output_list

    def _reset(self):
        self._current_batch_index = 0
