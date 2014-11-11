import os
import numpy as np

from bot_utils.dataset_readers import read_dir, natural_sort

class Caltech101DatasetReader(object): 
    """
    Dataset reader written to conform to scikit.data interface
    Attributes: 
      data:         [image_fn1, ...  ]
      target:       [class_id, ... ]
      target_ids:   [0, 1, 2, ..., 101]
      target_names: [car, bike, ... ]
    """
    def __init__(self, directory='', targets=None, num_targets=None, blacklist=['BACKGROUND_Google']): 
        self._dataset = read_dir(os.path.expanduser(directory), pattern='*.jpg')
        self._class_names = np.sort(self._dataset.keys())
        self._class_ids = np.arange(len(self._class_names), dtype=np.int)

        self.target_hash = dict(zip(self._class_names, self._class_ids))
        self.target_unhash = dict(zip(self._class_ids, self._class_names))

        # Only randomly choose targets if not defined
        if num_targets is not None and targets is None and \
           num_targets > 0 and num_targets < len(self._class_names): 
            inds = np.random.randint(len(self._class_names), size=num_targets)
            targets = self._class_names[inds]            
        else: 
            targets = self._class_names
        print 'Classes: %i' % len(targets)

        self.data, self.target = [], []
        for key, files in self._dataset.iteritems(): 
            if (targets is not None and key not in targets) or key in blacklist: 
                continue

            target_id = self.target_hash[key]
            self.data.extend(files)
            self.target.extend( [target_id] * len(files) )

        self.data = np.array(self.data)
        self.target = np.array(self.target)

        self.target_ids = sorted(np.unique(self.target))
        self.target_names = map(lambda tid: self.target_unhash[tid], self.target_ids)

