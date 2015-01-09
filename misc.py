import progressbar as pb

def setup_pbar(maxval): 
    widgets = ['Progress: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=maxval)
    pbar.start()
    pbar.increment = lambda : pbar.update(pbar.currval + 1)
    return pbar


# class IndexCounter(object): 
#     def __init__(self, start=0): 
#         self._idx = start

#     def increment(self): 
#         idx = np.copy(self._idx)
#         self._idx += 1 
#         return idx
