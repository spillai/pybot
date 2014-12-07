import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

global figures
figures = OrderedDict()

def imshow_plt(label, im, block=True):
    global figures
    if label not in figures: 
        figures[label] = plt.imshow(im, interpolation=None, animated=True, label=label)
        plt.tight_layout()
        plt.axis('off')
        
    figures[label].set_data(im)
    # figures[label].canvas.draw()
    # plt.draw()
    plt.show(block=block)

def bar_plt(label, ys, block=False):
    global figures
    if label not in figures:
        figures[label] = plt.bar(np.arange(len(ys)), ys, align='center')
        plt.title(label)
        # plt.tight_layout()
        # plt.axis('off')

    inds, = np.where([key == label for key in figures.keys()])
    for rect, y in zip(figures[label], ys): 
        rect.set_height(y)

    plt.draw()
    plt.show(block=block)


def imshow_cv(label, im, block=False, text=None): 
    vis = im.copy()
    if text is not None:
        cv2.rectangle(vis, (0, vis.shape[0]-18), (len(text) * 8, vis.shape[0]), (50, 50, 50), -1)
        cv2.putText(vis, '%s' % text, (2, vis.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                    (240, 240, 240), thickness = 1)
    cv2.imshow(label, vis)
    cv2.waitKey(0 if block else 1)

