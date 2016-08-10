import numpy as np

from pybot.utils.itertools_recipes import chunks
from sklearn.externals.joblib import Parallel, delayed

def parallel_for(iterable, func, batch_size, *func_args, **func_kwargs): 
    for chunk in chunks(iterable, batch_size): 
        res = Parallel(n_jobs=6, verbose=0) (
            delayed(func)(item, *func_args, **func_kwargs)
            for item in chunk )
        yield res

if __name__ == "__main__": 
    
    def get_iterable(): 
        idx = 0
        while idx < 50000: 
            idx += 1
            yield idx

    # for chunk in chunks(get_iterable(), 16): 
    #     print chunk 
    res = parallel_for(get_iterable(), np.sin, 16)
    for item in res: 
        print item
    
