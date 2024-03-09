import ctypes
import multiprocessing as mp

import numpy as np

from utils.medians_dataset import MediansDataset


def create_shared_cache(ctype, shape):
    size = np.prod(shape)
    shared_array_base = mp.Array(ctype, int(size))
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shape)
    return shared_array


class CachedMediansDataset(MediansDataset):
    def __init__(self, use_cache: bool, **kwargs):
        super().__init__(**kwargs)

        self.use_cache = use_cache
        if use_cache:
            self.medians_cache = create_shared_cache(ctypes.c_float, (len(self), *self.get_medians_shape()))
            self.labels_cache = create_shared_cache(ctypes.c_long, (len(self), *self.get_labels_shape()))
            self.epoch_passed = False

        self.logged_file = False
        self.logged_cache = False

    def __getitem__(self, idx):
        if self.use_cache:
            if not self.epoch_passed:
                if not self.logged_file:
                    print(f"Loading sample {idx} from file")
                    self.logged_file = True
                item = super().__getitem__(idx)
                self.medians_cache[idx] = item['medians']
                self.labels_cache[idx] = item['labels']
            else:
                if not self.logged_cache:
                    print(f"Loading sample {idx} from cache")
                    self.logged_cache = True
            return {'medians': self.medians_cache[idx], 'labels': self.labels_cache[idx]}
        else:
            return super().__getitem__(idx)
