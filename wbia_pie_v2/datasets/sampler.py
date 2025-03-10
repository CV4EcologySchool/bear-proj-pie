# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
import copy
import numpy as np
import utool as ut
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler


MAX_PER_SEQUENCE = 1


class RandomCopiesIdentitySampler(Sampler):
    """Randomly samples C copies of N identities each with K instances.
    Args:
        data_source (list): contains tuples of (img_path, pid, camid, dsetid)
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
        num_copies (int): number of copies of each example
    """

    def __init__(self, data_source, batch_size, num_instances, num_copies=1):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_copies = num_copies
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, items in enumerate(data_source):
            pid = items[1]
            seq_id = items[2]
            self.index_dic[pid].append((index, seq_id))
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            seq_ids = [idx[1] for idx in idxs]
            assert min(list(ut.dict_hist(seq_ids).values())) >= MAX_PER_SEQUENCE
            num = len(set(seq_ids)) * MAX_PER_SEQUENCE
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            assert len(set([idx[1] for idx in idxs])) >= self.num_instances, 'BAD BAD BAD'
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            ##
            seen_seq_ids = []
            for idx, seq_id in idxs:
                if seen_seq_ids.count(seq_id) >= MAX_PER_SEQUENCE:
                    continue
                seen_seq_ids.append(seq_id)
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
                ##
        
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(duplicate_list(batch_idxs, self.num_copies))
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def duplicate_list(a, k):
    """Duplicate each element in list a for k times"""
    return [val for val in a for _ in range(k)]
