import numpy as np

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
        fn, class_names = zip(*map(lambda x: (x['filename'], x['name']), dataset.meta))
        self._class_names = list(set(class_names))
        self._class_ids = np.arange(len(self._class_names), dtype=np.int)

        self.target_hash = dict(zip(self._class_names, self._class_ids))
        self.target_unhash = dict(zip(self._class_ids, self._class_names))
        class_ids = map(lambda cname: self.target_hash[cname], class_names)

        # Options: 
        # 1. Select full set: targets = None, 
        # 2. Specific set:    targets = ['a', 'b']
        # 3. Target count:    targets = 10 (randomly pick 10 classes)

        # Pick full/specified dataset            
        if targets is None: 
            target_ids = self._class_ids
            self.data, self.target = fn, map(lambda cname: self.target_hash[cname], class_names)

        # Or specific count/set
        else: 
            # If integer valued, retrieve targets
            if isinstance(targets, int) and targets < len(self._class_names): 
                inds = np.random.randint(len(self._class_names), size=targets)
                target_ids = self._class_ids[inds]

            # If targets are list of strings
            elif isinstance(targets, list) and len(targets) < len(self._class_names): 
                # Print error if target not in _class_names
                for t in targets: 
                    if t not in self._class_names: 
                        raise ValueError('Target %s not in class_names' % t)
                target_ids = map(lambda cname: self.target_hash[cname], targets)
            else: 
                raise ValueError('targets are not list of strings or integer')

            # Filter data to specific targets_set
            target_ids_set = set(target_ids)
            self.data, self.target = zip(*filter(lambda x: x[1] in target_ids_set, zip(fn, class_ids)))


        self.data = np.array(self.data)
        self.target = np.array(self.target)

        self.target_ids = sorted(np.unique(self.target))
        self.target_names = map(lambda tid: self.target_unhash[tid], self.target_ids)



