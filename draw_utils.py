import cv2
import numpy as np

from bot_utils.plot_utils import colormap
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
        # cv2.circle(out, tuple(map(int, pt)), size, tuple(col), -1, lineType=cv2.CV_AA)
    return out

def draw_lines(im, pts1, pts2, colors=None, thickness=1): 
    out = to_color(im)
    if colors is not None: 
        cols = colors.astype(np.int64)
    else: 
        cols = np.tile([0, 255, 0], (len(pts1), 1)).astype(np.int64)

    for col, pt1, pt2 in zip(cols, pts1, pts2): 
        cv2.line(out, (pt1[0], pt1[1]), (pt2[0], pt2[1]), tuple(col), thickness)
    return out

def draw_matches(out, pts1, pts2, colors=None, thickness=1, size=2): 
    out = draw_lines(out, pts1, pts2, colors=colors, thickness=thickness)
    out = draw_features(out, pts2, colors=colors, size=size)
    return out

def draw_bboxes(vis, bboxes, texts=None, ellipse=False, colored=True):
    if not len(bboxes): 
        return vis

    if not colored: 
        cols = np.tile([240,240,240], [len(bboxes), 1])
    else: 
        N = 20
        cwheel = colormap(np.linspace(0, 1, N))
        cols = np.vstack([cwheel[idx % N] for idx, _ in enumerate(bboxes)])            

    texts = [None] * len(bboxes) if texts is None else texts
    for col, b, t in zip(cols, bboxes, texts): 
        if ellipse: 
            cv2.ellipse(vis, ((b[0]+b[2])/2, (b[1]+b[3])/2), ((b[2]-b[0])/2, (b[3]-b[1])/2), 0, 0, 360, 
                        color=tuple(col), thickness=1)
        else: 
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), tuple(col), 2)
        if t: 
            annotate_bbox(vis, b, title=t)
    return vis

def annotate_bbox(vis, coords, color=(0,200,0), title=''): 
    # Bounding Box and top header
    icoords = coords.astype(np.int32)
    cv2.rectangle(vis, (icoords[0], icoords[1]), (icoords[2], icoords[3]), color, 2)
    # cv2.rectangle(vis, (icoords[0]-1, icoords[1]-15), (icoords[2]+1, icoords[1]), color, -1)
    cv2.putText(vis, '{}'.format(title), (icoords[0], icoords[1]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), thickness=1, lineType=cv2.CV_AA)
    return vis

def annotate_bboxes(vis, bboxes, texts, colors=None):
    # target_name.title().replace('_', ' '))
    for bbox,text in izip(bboxes, texts): 
        box_color = (0, 200, 0) # if UWRGBDDataset.get_category_name(target) != 'background' else (100, 100, 100)
        annotate_bbox(vis, bbox, color=box_color, title=text)
    return vis
        
def draw_ellipses(im, ellipses): 
    for e in ellipses:
        cv2.ellipse(im, e, (255, 255, 0) if im.ndim == 3 else 255,1)
    return im

def draw_hulls(im, hulls): 
    assert(isinstance(hulls, list))
    cv2.polylines(im, hulls, 1, (0, 255, 0) if im.ndim == 3 else 255, thickness=1)       
    return im

