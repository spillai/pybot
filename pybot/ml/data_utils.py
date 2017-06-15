import numpy as np
from collections import deque, defaultdict
from pybot.utils.itertools_recipes import chunks

def concat_chunked_dicts(dlist):
    """
    Concatenate individual arrays in dictionary
       TODO: defaultdict is the right way to do it, except for
       conversion to dict in the final return call. Keras requires 
       type as dict
    """
    batch = defaultdict(list)
    for item in dlist:
        for k,v in item.iteritems():
            batch[k].append(v)
    for k,v in batch.iteritems():
        batch[k] = np.concatenate(v)
    return dict(batch)

def chunked_data(iterable, batch_size=10):
    """
    For tuples: 
       arg = ([np.array, np.array], {'output': np.array})
    

    For dictionaries: 
       arg = ({'input': np.array, 'input2': np.array}, {'output': np.array})
    """
    for batch in chunks(iterable, batch_size):
        args = zip(*batch)

        # (arg), (arg), ...
        # arg = ([x1,x2], y) 
        # type(args[0]) = tuple
        # type(args[0][0]) = list
        
        if isinstance(args[0][0], dict):
            items = [concat_chunked_dicts(arg) for arg in args]
            
        elif isinstance(args[0][0], np.ndarray):
            items = [np.concatenate(arg) for arg in args]

        elif isinstance(args[0][0], list) and isinstance(args[0][0][0], np.ndarray):
            items = [[np.concatenate(item) for item in arg] for arg in args]

        else:
            raise TypeError('''Unknown type: either dict, np.array, or list of np.arrays can be batched'''
                            '''Type is {}'''.format(type(args[0][0])))
            
        yield tuple(items)
