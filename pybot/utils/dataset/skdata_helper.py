# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
from itertools import izip
from pybot.utils.db_utils import AttrDict

class skdataHelper(object): 
    """
    skdata helper class written to conform to scikit.data interface

    Attributes: 
      data:         [image_fn1, ...  ]
      target:       [class_id, ... ]
      target_ids:   [0, 1, 2, ..., 101]
      target_names: [car, bike, ... ]
    """
    def __init__(self, dataset, targets=None): 
        self._dataset = dataset
        _ = self._dataset.meta

        self._class_names = np.array(self._dataset.names)
        self._class_ids = np.arange(len(self._class_names), dtype=np.int)

        self.target_hash = dict(zip(self._class_names, self._class_ids))
        self.target_unhash = dict(zip(self._class_ids, self._class_names))

        # Options: 
        # 1. Select full set: targets = None, 
        # 2. Specific set:    targets = ['a', 'b']
        # 3. Target count:    targets = 10 (randomly pick 10 classes)

        # Pick full/specified dataset
        if targets is None: 
            self._whitelist = None

        # Or specific count/set
        else: 
            # If integer valued, retrieve targets
            if isinstance(targets, int) and targets < len(self._class_names): 
                inds = np.random.randint(len(self._class_names), size=targets)
                self._whitelist = set(self._class_names[inds])

            # If targets are list of strings
            elif isinstance(targets, list) and len(targets) < len(self._class_names): 
                # Print error if target not in _class_names
                for t in targets: 
                    if t not in self._class_names: 
                        raise ValueError('Target %s not in class_names' % t)
                self._whitelist = set(targets)
            else: 
                raise ValueError('targets are not list of strings or integer')

        self.target_names = np.array(list(self._whitelist)) if self._whitelist is not None else self._class_names 
        self.target_ids = map(lambda cname: self.target_hash[cname], self.target_names)

    def _split(self, split=None): 
        if split:
            inds = self._dataset.splits[split]
        else:
            inds = xrange(len(self._dataset.meta))
            
        data = [ (self._dataset.meta[ind]['filename'], self.target_hash[self._dataset.meta[ind]['name']])
                 for ind in inds 
                 if (self._whitelist is None) or (self._dataset.meta[ind]['name'] in self._whitelist) ]

        fn, target = zip(*data)
        return np.array(fn), np.array(target, dtype=np.int32)

    def _prepare_dataset(self, X, y): 
        return (AttrDict(filename=x_t, target=y_t) for (x_t, y_t) in izip(X, y))

    def get_train_test_split(self): 
        for split_idx in range(0, self._dataset.num_splits): 
            X_train, y_train = self._split(split='train_%i' % split_idx)
            X_test, y_test = self._split(split='test_%i' % split_idx)
            return self._prepare_dataset(X_train, y_train), self._prepare_dataset(X_test, y_test)



