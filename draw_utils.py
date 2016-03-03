import cv2
import numpy as np
from bot_vision.image_utils import to_color

def draw_features(im, pts, colors=None, size=2): 
    out = to_color(im)
    if colors is not None: 
        cols = colors.astype(np.int64)
    else: 
        cols = np.tile([0, 255, 0], (len(pts), 1)).astype(np.int64)

    for col, pt in zip(cols, pts): 
        tl = np.int32(pt - size)
        br = np.int32(pt + size)
        cv2.rectangle(out, (tl[0], tl[1]), (br[0], br[1]), tuple(col), -1)
        # cv2.circle(out, tuple(map(int, pt)), 3, (0,255,0), -1, lineType=cv2.CV_AA)
    return out
