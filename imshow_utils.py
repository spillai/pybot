import cv2
import numpy as np
import matplotlib.pyplot as plt

global figures
figures = {}

def imshow_plt(label, im, block=True):
    global figures
    if label not in figures: 
        figures[label] = plt.imshow(im, interpolation=None, animated=True, label=label)
        plt.tight_layout()
        plt.axis('off')
        
    figures[label].set_data(im)
    figures[label].canvas.draw()
    # plt.draw()
    plt.show(block=block)

def bar_plt(label, ys, block=False):
    global figures
    if label not in figures: 
        figures[label] = plt.bar(np.arange(len(ys)), ys, align='center')
        # plt.tight_layout()
        # plt.axis('off')
        
    for rect, y in zip(figures[label], ys): 
        rect.set_height(y)

    plt.draw()
    plt.show(block=block)


def imshow_cv(label, im): 
    cv2.imshow(label, im)
    cv2.waitKey(1)
