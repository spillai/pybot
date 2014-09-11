import cv2
import numpy as np

def dense_optical_flow(im1, im2, pyr_scale=0.5, levels=3, winsize=15, iterations=5, poly_n=1.2, poly_sigma=0): 
    return cv2.calcOpticalFlowFarneback(im1, im2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma)

def sparse_optical_flow(im1, im2, pts, fb_threshold=0, 
                        window_size=15, max_level=2, 
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)): 
    p1, st, err = cv2.calcOpticalFlowPyrLK(im1, im2, pts, None, 
                                           winSize=(window_size, window_size), 
                                           maxLevel=max_level, criteria=criteria )

    # if fb_threshold > 0:     
    #     p0r, st, err = cv2.calcOpticalFlowPyrLK(im2, im1, p1, None, 
    #                                        winSize=(window_size, window_size), 
    #                                        maxLevel=max_level, criteria=criteria)
    #     d = abs(p0-p0r).reshape(-1, 2).max(-1)
    #     good = d < fb_threshold

    return p1, st, err

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    m = np.bitwise_and(np.isfinite(fx), np.isfinite(fy))
    lines = np.vstack([x[m], y[m], x[m]+fx[m], y[m]+fy[m]]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def test_flow(img1, img2): 
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, 0.5, 3, 15, 3, 5, 1.2, 0)
    cv2.imshow('flow', draw_flow(gray1, flow))
    cv2.imshow('flow HSV', draw_hsv(flow))
    cv2.imshow('warp', warp_flow(cur_glitch, flow))

