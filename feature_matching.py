import cv2
import numpy as np
from itertools import chain, izip

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)
    # def onmouse(event, x, y, flags, param):
    #     cur_vis = vis
    #     if flags & cv2.EVENT_FLAG_LBUTTON:
    #         cur_vis = vis0.copy()
    #         r = 8
    #         m = (np.linalg.norm(p1 - (x, y)) < r) | (np.linalg.norm(p2 - (x, y)) < r)
    #         idxs = np.where(m)[0]
    #         kp1s, kp2s = [], []
    #         for i in idxs:
    #              (x1, y1), (x2, y2) = p1[i], p2[i]
    #              col = (red, green)[status[i]]
    #              cv2.line(cur_vis, (x1, y1), (x2, y2), col)
    #              kp1, kp2 = kp_pairs[i]
    #              kp1s.append(kp1)
    #              kp2s.append(kp2)
    #         cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
    #         cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)

    #     cv2.imshow(win, cur_vis)
    # cv2.setMouseCallback(win, onmouse)
    return vis


class RichFeatureMatching: 
    def __init__(self, detector='SIFT', descriptor='SIFT'): 
        self.detector = cv2.FeatureDetector_create(detector)
        self.extractor = cv2.DescriptorExtractor_create(descriptor)
        self.matcher = cv2.DescriptorMatcher_create('FlannBased')

    def match(self, im1, im2, mask1=None, mask2=None):
        # Feature detection
        kpts1 = self.detector.detect(im1, mask=mask1)
        kpts2 = self.detector.detect(im2, mask=mask2)
        
        # Extract descriptors
        kpts1, desc1 = self.extractor.compute(im1, kpts1)
        kpts2, desc2 = self.extractor.compute(im2, kpts2)

        # Match
        m12 = self.matcher.knnMatch(desc1, desc2, 1)
        m21 = self.matcher.knnMatch(desc2, desc1, 1)
        
        # FWD-BWD check
        m12, m21 = list(chain(*m12)), list(chain(*m21))
        fb_matches = [ m12_item 
                       for m12_item in m12 
                       if m12_item.queryIdx == m21[m12_item.trainIdx].trainIdx ]

        # Re-assign kpts to valid ones
        kpts1 = [ kpts1[m12_item.queryIdx] for m12_item in fb_matches ]
        kpts2 = [ kpts2[m12_item.trainIdx] for m12_item in fb_matches ]

        pts1 = np.vstack([ kp.pt for kp in kpts1 ]).astype(np.float32)
        pts2 = np.vstack([ kp.pt for kp in kpts2 ]).astype(np.float32)

        # Fundamental matrix
        # M, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        F, status = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 
                                             1.0, 0.998)

        # Inliers matches
        inliers = np.vstack([ np.array([m12_item.queryIdx, m12_item.trainIdx])
                              for m12_item in fb_matches ]).astype(np.int32)
        inliers = inliers[status.ravel() == 1]

        explore_match('win', im1, im2, zip(kpts1, kpts2), status)
        cv2.waitKey(0)

if __name__ == "__main__": 
    im1 = cv2.imread('box.png', 0)
    im2 = cv2.imread('box_in_scene.png', 0)
    RichFeatureMatching().match(im1, im2)
