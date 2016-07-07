import cv2
import numpy as np

def convex_hull(pts, ccw=True): 
    """
    Returns the convex hull of points, ordering them in ccw/cw fashion

    Note: Since the orientation of the coordinate system is x-right,
    y-up, ccw is interpreted as cw in the function call.

    """ 
    assert(pts.ndim == 2 and pts.shape[1] == 2)
    return (cv2.convexHull(pts.reshape(-1,1,2), clockwise=ccw)).reshape(-1,2)
