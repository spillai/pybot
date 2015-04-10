import cv2
import sys
import time
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

def print_status(vis, text=None): 
    if text is not None:
        cv2.rectangle(vis, (0, vis.shape[0]-18), (len(text) * 8, vis.shape[0]), (50, 50, 50), -1)
        cv2.putText(vis, '%s' % text, (2, vis.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                    (240, 240, 240), thickness = 1)

def imshow_cv(label, im, block=False, text=None): 
    vis = im.copy()
    print_status(vis, text=text)
    cv2.imshow(label, vis)
    ch = cv2.waitKey(0 if block else 1) & 0xFF
    if ch == ord(' '):
        cv2.waitKey(0)
    if ch == ord('s'):
        fn = 'img-%s.png' % time.strftime("%Y-%m-%d-%H-%M-%S")
        print 'Saving %s' % fn
        cv2.imwrite(fn, vis)
    elif ch == 27 or ch == ord('q'):
        sys.exit(1)

def annotate_bbox(vis, bbox, color=(0,200,0), title=''): 
    # Bounding Box and top header
    cv2.rectangle(vis, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), color, 2)
    cv2.rectangle(vis, (bbox['left']-1, bbox['top']-15), (bbox['right']+1, bbox['top']), color, -1)

    cv2.putText(vis, '%s' % title, (bbox['left'], bbox['top']-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), thickness=1, lineType=cv2.CV_AA)

