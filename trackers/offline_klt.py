'''
Lucas-Kanade tracker

@author: Sudeep Pillai (Last Edited: 01 May 2014)

Notes: 

a. Forward-Backward OF PyramidLK error 
b. Gating of flows
====================

'''

import cv2, time, os.path
import numpy as np

from collections import namedtuple, deque
# from sklearn.neighbors import BallTree
from bot_utils.db_utils import AttrDict

# from utils.data_containers import Feature3DData
# from utils.optflow_utils import draw_flow, draw_hsv
# import utils.draw_utils as draw_utils

from .base_klt import BaseKLT
from bot_vision.imshow_utils import imshow_cv
from bot_vision.image_utils import to_color, to_gray, gaussian_blur
from bot_utils.plot_utils import colormap

from bot_utils.itertools_recipes import pairwise

class FwdBwdKLT2(BaseKLT): 
    """
    Offline KLT Tracker using full information    
    """
    fwdbwd_params = AttrDict(overlap_threshold=5, error_threshold=3.0)
    default_params = AttrDict(
        BaseKLT.default_params, 
        fwdbwd_params=fwdbwd_params
    )

    def __init__(self, dataset, params=default_params):
        BaseKLT.__init__(self, params=params)
        self.params = params.fwdbwd_params
        
        # Copy frame iterator
        self.frames = dataset.iteritems
        self.nframes = dataset.length

        # Process frames provided
        st = time.time()
        self.run()
        self.log.info('Processed %i frames in %f s' % (self.nframes, time.time() - st))


    def draw_tracks(self, tpts, im): 
        T, N = tpts.shape[:2]
        for tid in np.arange(N): 
            pts = tpts[:,tid,:]
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]
            cv2.polylines(im,[np.vstack(pts).astype(np.int32)], False, 
                              tuple(map(int, colormap(tid % 20 / 20.0).ravel())), 
                              thickness=1, lineType=cv2.CV_AA)

            
    def draw_good_tracks(self, im): 
        T, N = self.fpts.shape[:2]
        assert(self.fpts.shape == self.bpts.shape)
        errs = np.nanmean(np.linalg.norm(self.fpts - self.bpts, axis=2), axis=0)
        counts = np.sum(np.bitwise_and(np.isfinite(self.fpts[:,:,0]), np.isfinite(self.bpts[:,:,0])), axis=0)
        inds, = np.where(counts > self.params.overlap_threshold)

        for err, count, tid in zip(errs[inds], counts[inds], inds): 
            pts = self.fpts[:,tid,:]
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]
            cv2.polylines(im,[np.vstack(pts).astype(np.int32)], False, 
                              tuple(map(int, colormap(err/10).ravel())), 
                              thickness=1, lineType=cv2.CV_AA)

    def get_features(self, Xs, Ns): 
        T, N = self.fpts.shape[:2]
        errs = np.nanmean(np.linalg.norm(self.fpts - self.bpts, axis=2), axis=0)
        counts = np.sum(np.bitwise_and(np.isfinite(self.fpts[:,:,0]), np.isfinite(self.bpts[:,:,0])), axis=0)
        valid_ninds, = np.where(np.bitwise_and(counts > self.params.overlap_threshold, 
                                               errs < self.params.error_threshold))        

        from bot_utils.data_containers import Feature3DData

        # Retrieve 3D
        for tidx, (Xidx, Nidx, pts) in enumerate(izip(Xs, Ns, self.fpts)): 
            # Find valid inds that are finite
            ninds = valid_ninds[np.where(np.isfinite(pts[valid_ninds]).all(axis=1))[0]]
            xys = pts[ninds].astype(int)

            # Copy 2D data
            data.xy[ninds,tidx] = pts[ninds]

            try: 
                # Copy 3D data
                data.xyz[ninds,tidx] = Xidx[xys[:,1],xys[:,0],:]

                # # Copy Normals data
                # data.normal[ninds,tidx] = Nidx[xys[:,1],xys[:,0],:]

                ninds_ = ninds[np.isfinite(data.xyz[ninds,tidx]).all(axis=1)] #  &
                               # np.isfinite(data.normal[ninds,tidx]).all(axis=1)]
                data.idx[ninds_,tidx] = 1
            except IndexError: 
                pass

            # # Mark outliers
            # outliers = ninds[np.where((np.fabs(self.fpts[tidx, ninds] - 
            #                                    self.bpts[tidx, ninds]) > 1.0).any(axis=1))[0]]
            # data.idx[outliers,tidx] = -1
        draw_utils.publish_point_cloud('tracks3d', data.xyz.reshape(-1,3), c='b')
        return data 



    def get_tracks(self): 
        T, N = self.fpts.shape[:2]
        errs = np.nanmean(np.linalg.norm(self.fpts - self.bpts, axis=2), axis=0)
        counts = np.sum(np.bitwise_and(np.isfinite(self.fpts[:,:,0]), np.isfinite(self.bpts[:,:,0])), axis=0)
        inds, = np.where(counts > self.params.overlap_threshold)

        for err, count, tid in zip(errs[inds], counts[inds], inds): 
            pts = self.fpts[:,tid,:]
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]        
        
    def visualize_tracks(self, name, im, pts): 
        vis = to_color(im)
        self.draw_tracks(pts, vis)
        return vis

    def visualize(self, fvis, bvis): 
        imshow_cv('FWD/BWD', np.vstack([fvis, bvis]))
        imshow_cv('FWD/BWD CHECK', fvis/2 + bvis/2)

    def visualize_error(self, im): 
        errvis = to_color(im)
        self.draw_good_tracks(errvis)
        imshow_cv('FWD/BWD ERR', errvis)
        
    def run(self): 

        # Store fwd/bwd features
        T = self.nframes
        ims = deque(maxlen=2)

        # I. FWD-Flow Extraction
        print 'Forward flow ====================================> '
        for tidx, im in enumerate(self.frames()): 
            ims.append(im)

            # 1. Detect/Track
            if len(ims) != 2: 
                # a. Feature Extraction with first frame
                pts = self.detector.process(ims[-1], mask=None)
                ids = None
                prune = False

                # b. Initialize data matrix (T x N x 2)
                N = len(pts)
                self.fpts = (np.empty(shape=(T, N, 2)) * np.nan).astype(np.float32)
                self.bpts = (np.empty(shape=(T, N, 2)) * np.nan).astype(np.float32)

            else: 
                # a. Track object
                ids, ppts = self.tm.ids, self.tm.pts
                pts = self.tracker.track(ims[-2], ims[-1], ppts)
                prune = True

            # 2. Add tracks
            self.tm.add(pts, ids=ids, prune=prune)

            # 3. Save forward flow features
            try: 
                ids, pts = self.tm.ids, self.tm.pts
                self.fpts[tidx,ids,:] = pts
            except: 
                pass

        print 'Forward flow DONE -------------------------------> '
        fvis = self.visualize_tracks('FWD', ims[-1], self.fpts)

        # I. BWD-Flow Extraction
        print 'Backward flow ====================================> '
        detected = set([])
        ims = deque(maxlen=2)
        self.tm.reset()
        for _tidx, im in enumerate(self.frames(reverse=True)): 
            tidx = T-1-_tidx
            ims.append(im)

            # Only look at pairs
            if len(ims) != 2: 
                continue

            # 1. Identify track and detect ids, points
            # Detect when only latter frame is finite
            detect_ids_, = np.where(np.isfinite(self.fpts[tidx+1,:,0]))

            # Detect when both frames are finite (the latter detection is finicky)
            # detect_ids_, = np.where(np.bitwise_and(np.isfinite(self.fpts[tidx,:,0]), 
            #                                        np.isfinite(self.fpts[tidx+1,:,0])))

            # 2. Add to tracker and save
            # Only add detections if not previously detected
            detect_ids = np.array(list(set(detect_ids_) - detected))
            detected.update(detect_ids)
            if len(detect_ids): 
                detect_pts = self.fpts[tidx, detect_ids]
                self.tm.add(detect_pts, ids=detect_ids, prune=False)

                ids, pts = self.tm.ids, self.tm.pts
                # print 'tm: %i' % len(self.tm.ids)
                self.bpts[tidx,ids,:] = pts

            # 3. Track existing features
            # Look for detections followed by un-tracked feature
            track_ids, = np.where(np.isfinite(self.bpts[tidx+1,:,0]))
            if len(track_ids): 
                track_ppts = self.bpts[tidx+1, track_ids]
                track_pts = self.tracker.track(ims[-2], ims[-1], track_ppts)

                self.tm.add(track_pts, ids=track_ids, prune=True)
                ids, pts = self.tm.ids, self.tm.pts
                # print 'tm: %i' % len(self.tm.ids)
                self.bpts[tidx,ids,:] = pts

            # print 'detect: %i' % (len(detect_ids))
            # print 'tidx: %i, detect: %i track: %i' % (tidx, len(detect_ids), len(track_ids))

        print 'Forward flow DONE -------------------------------> '
        bvis = self.visualize_tracks('BWD', ims[-1], self.bpts)

        print 'Final visualization'
        self.visualize(fvis, bvis)
        self.visualize_error(ims[-1])
        cv2.waitKey(0)

        # Tracks
        self.get_tracks()

class OfflineKLT(BaseKLT): 

    oklt_params_ = AttrDict( err_th = 1.0 )
    flow_info = namedtuple('flow_info', ['flow', 'status', 'err'])

    def __init__(self, images, oklt_params=oklt_params_, **kwargs):
        BaseKLT.__init__(self, **kwargs)

        # # OKLT Params
        # self.oklt_params = oklt_params
        # self.log.debug('Offline KLT Params: %s' % oklt_params)

        # Images indexed
        self.images = map(lambda x: self.preprocess_im(x), images)

    def ffarneback(self, p0init, idx_from, idx_to): 
        flow = cv2.calcOpticalFlowFarneback(self.images[idx_from], self.images[idx_to], 
                                            None, **self.farneback_params_) # 0.5, 3, 15, 3, 5, 1.2, 0
        flow = cv2.medianBlur(flow, 5)

        # cv2.imshow('flow', draw_hsv(flow))
        # cv2.waitKey(10)
        
        valid = np.isfinite(p0init).all(axis=1)
        vinds = np.where(valid)[0]

        xys = p0init[valid].astype(int)
        fflow = flow[xys[:,1], xys[:,0]]

        valid_flow = np.isfinite(fflow).all(axis=1)
        flow_inds = np.where(valid_flow)[0]

        inds = vinds[flow_inds]

        # Retain flow info 
        return AttrDict(flow=fflow[inds], inds=inds)

    def flk(self, p0init, idx_from, idx_to): 
        # Forward flow
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(self.images[idx_from], self.images[idx_to], 
                                               p0init, None, **self.lk_params)
        # Invalidate pts without good flow
        valid = (st1 != 0)
        inds = np.where(valid)[0]
        ninds = np.where(~valid)[0]
        p1[ninds,:] = np.nan

        # Retain flow info 
        return AttrDict(flow=p1-p0init, inds=inds, err=err1)
        
    def fblk(self, p0init, idx_from, idx_to): 
        # Forward flow
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(self.images[idx_from], self.images[idx_to], 
                                               p0init, None, **self.lk_params)
        # Invalidate pts without good flow
        inds1 = np.where(st1 == 0)[0]
        p1[inds1,:] = np.nan

        # Backward flow
        p0, st0, err0 = cv2.calcOpticalFlowPyrLK(self.images[idx_to], self.images[idx_from], 
                                               p1, None, **self.lk_params)
        # Invalidate pts without good flow
        inds0 = np.where(st0 == 0)[0]
        p0[inds0,:] = np.nan

        # Forward-backward error
        fberr = p0init - p0 
        # print 'err: ', self.oklt_params.err_th, fberr

        # If either axis has high error, remove pt
        valid = (fberr < self.oklt_params.err_th).all(axis=1)
        inds = np.where(valid)[0]

        ninds = np.where(~valid)[0]
        p1[ninds,:] = np.nan

        # Retain flow info 
        return AttrDict(flow=p1-p0init, inds=inds, err=fberr)

# class OfflineStandardKLT(OfflineKLT):
#     def __init__(self, *args, **kwargs):
#         OfflineKLT.__init__(self, *args, **kwargs)

#         # Process frames provided
#         st = time.time()
#         self.run()
#         self.log.debug('Processed %i frames in %f s' % (len(self.images), time.time() - st))

#     def run(self): 
#         # 1. Feature Extraction with first frame
#         pts = self.process_im(self.images[0], mask=None)

#         # 2. Initialize data matrix (T x N x 2)
#         T, N = len(self.images), len(pts)
#         self.pts = (np.empty(shape=(T, N, 2)) * np.nan).astype(np.float32)

#         # Init forward features
#         self.pts[0] = pts

#         # 3. FWD/BWD-Flow Extraction: For (1,2), (2,3), ....
#         finds = np.arange(0, T)
#         pair_inds = zip(finds[:-1], finds[1:])
#         self.log.debug('Forward flow ========> %s ' % pair_inds)
#         for (idx0,idx1) in pair_inds: 
#             # FWD-flow: f_flow[1] = lk(1,2)
#             if self.oklt_params.FB_check: 
#                 fflow0 = self.fblk(self.pts[idx0], idx0, idx1)
#             else: 
#                 fflow0 = self.flk(self.pts[idx0], idx0, idx1)
            
#             # Apply forward predictions for valid flow regions
#             self.pts[idx1,fflow0.inds] = \
#                 self.pts[idx0,fflow0.inds] + fflow0.flow[fflow0.inds]

#     def get_feature_data(self, Xs, Ns=None): 
#         T, N, _ = self.pts.shape

#         # Create feature data container
#         data = Feature3DData.empty(N, T)

#         # Retrieve 3D
#         for tidx,pts in enumerate(self.pts): 
#             # Find valid inds that are finite
#             ninds = np.where(np.isfinite(pts).all(axis=1))[0]
#             xys = pts[ninds].astype(int)

#             # Copy 2D data
#             data.xy[ninds,tidx] = pts[ninds]

#             try: 
#                 # Copy 3D data
#                 data.xyz[ninds,tidx] = Xs[tidx][xys[:,1],xys[:,0],:]

#                 # Copy Normals data
#                 if Ns is not None: 
#                     data.normal[ninds,tidx] = Ns[tidx][xys[:,1],xys[:,0],:]

#                 ninds_ = ninds[np.isfinite(data.xyz[ninds,tidx]).all(axis=1) &
#                                np.isfinite(data.normal[ninds,tidx]).all(axis=1)]
#                 data.idx[ninds_,tidx] = 1
#             except IndexError: 
#                 pass

#             # # Mark outliers
#             # outliers = ninds[np.where((np.fabs(self.fpts[tidx, ninds] - 
#             #                                    self.bpts[tidx, ninds]) > 1.0).any(axis=1))[0]]
#             # data.idx[outliers,tidx] = -1

#         draw_utils.publish_point_cloud('tracks3d', data.xyz.reshape(-1,3), c='b')
#         return data 

    

class FwdBwdKLT(OfflineKLT): 
    def __init__(self, *args, **kwargs):
        OfflineKLT.__init__(self, *args, **kwargs)

        # Flow data
        self.f_flow, self.b_flow = dict(), dict()

        # Process frames provided
        st = time.time()
        self.run()
        self.log.info('Processed %i frames in %f s' % (len(self.images), time.time() - st))

    def run(self): 
        # 1. Feature Extraction with first frame
        pts = self.process_im(self.images[0], mask=None)
 
        # 2. Initialize data matrix (T x N x 2)
        T, N = len(self.images), len(pts)
        self.fpts = (np.empty(shape=(T, N, 2)) * np.nan).astype(np.float32)
        self.bpts = self.fpts.copy()
        
        # Init forward features
        self.fpts[0] = pts

        # 3. FWD/BWD-Flow Extraction: For (1,2), (2,3), ....
        finds = np.arange(0, T)
        pair_inds = zip(finds[:-1], finds[1:])
        self.log.debug('Forward flow ========> %s' % pair_inds)
        for (idx0,idx1) in pair_inds: 
            # FWD-flow: f_flow[1] = lk(1,2)
            if self.oklt_params.OF_method == 'dense': 
                fflow0 = self.ffarneback(self.fpts[idx0], idx0, idx1)
            elif self.oklt_params.OF_method == 'lk': 
                if self.oklt_params.FB_check: 
                    fflow0 = self.fblk(self.fpts[idx0], idx0, idx1)
                else: 
                    fflow0 = self.flk(self.fpts[idx0], idx0, idx1)
                self.f_flow[idx0] = fflow0
            else: 
                raise RuntimeError('Unknown OF_method: %s' % self.oklt_params.OF_method)
            
            # Apply forward predictions for valid flow regions
            self.fpts[idx1,fflow0.inds] = \
                self.fpts[idx0,fflow0.inds] + fflow0.flow[fflow0.inds]

        
        # 4. Last valid fwd-prediction = First valid bwd-prediction
        rinds = finds[::-1]
        marked = np.array([True] * N)
        for ridx in rinds: 
            # Find valid forward predictions
            fvalid = np.isfinite(self.fpts[ridx]).all(axis=1)
            
            # Check if has been considered valid already
            mark_inds = np.where(fvalid & marked)[0]
            marked[mark_inds] = False
            # print ridx, mark_inds

            # Last valid FWD prediction := First valid BWD prediction
            self.bpts[ridx, mark_inds] = self.fpts[ridx, mark_inds]

        # 5. BWD flow
        pair_rinds = zip(rinds[:-1], rinds[1:])
        self.log.debug('Backward flow ========> %s' % pair_rinds)
        for (idx1,idx0) in pair_rinds: 
            # BWD-flow: b_flow[2] = lk(2,1)
            bflow1 = self.fblk(self.bpts[idx1], idx1, idx0)
            self.b_flow[idx1] = bflow1

            # Make sure nans are not written over bpts
            bvalid = ((np.isfinite(self.bpts[idx1])).all(axis=1) & 
                      (np.isnan(self.bpts[idx0])).all(axis=1) & 
                      (np.isfinite(bflow1.flow)).all(axis=1)) 

            bvalid_inds = np.where(bvalid)[0]
            self.bpts[idx0,bvalid_inds] = \
                self.bpts[idx1,bvalid_inds] + bflow1.flow[bvalid_inds]

    # def subpix_pts(self, X, pts): 
    #     # Ensure within image bounds
    #     pinds = (pts > 5).all(axis=1) & (pts[:,0] < 635) & (pts[:,1] < 475)
    #     if not len(pinds): 
    #         return np.array([])
    #     pts = pts[pinds]

    #     ptsf = np.floor(pts)
    #     ptsc = np.ceil(pts)

    #     # Barycentric coords
    #     finds = np.fabs(ptsf - pts) <= 0.01
    #     cinds = np.fabs(ptsc - pts) <= 0.01
    #     ptsf[finds] = ptsf[finds] - 1
    #     ptsc[cinds] = ptsc[cinds] + 1

    #     for pt, ptf, ptc in zip(pts, ptsf, ptsc): 
    #         pass

    #     print np.hstack([pts, ptsf, ptsc])

    def viz_pts(self, Xs): 
        X = []
        
        avg_err = np.sqrt(np.nanmean(np.sum((self.fpts - self.bpts) ** 2, axis=2), axis=0))
        valid = avg_err < 5.0
        valid_inds = np.where(valid)[0]
        # print 'Average error: ', avg_err

        for idx,pts in enumerate(self.fpts): 

            # Ensure low-fb-error
            # err = self.fpts[idx] - self.bpts[idx]
            # valid =  (err < self.oklt_params.err_th).all(axis=1)
            # valid_inds = np.where(valid)[0]

            # Plot 
            rgb = cv2.cvtColor(self.images[idx].copy(), cv2.COLOR_GRAY2BGR)            
            for v,pt in zip(valid, pts): 
                if np.isnan(pt).any(): continue
                cv2.circle(rgb, tuple(map(int, pt)), 
                           2, (0,255,0) if v else (0,0,255), -1, lineType=cv2.LINE_AA)

            # Plot Trajectories in 3d
            yx = pts[valid_inds]
            yxs = yx[np.where(np.isfinite(yx))[0]].astype(int)
            try: 
                X.append(Xs[idx][yxs[:,1],yxs[:,0],:])
            except IndexError: 
                pass

            cv2.imshow('rgb', rgb)
            cv2.waitKey(30)
            # time.sleep(0.2)

        draw_utils.publish_point_cloud('tracks', np.vstack(X), c='b')

    def get_feature_data(self, Xs, Ns=None): 
        T, N, _ = self.fpts.shape

        # Create feature data container
        data = Feature3DData.empty(N, T)

        # Pick tracks with small average error
        avg_err = np.sqrt(np.nanmean(np.sum((self.fpts - self.bpts) ** 2, axis=2), axis=0))
        valid = avg_err < 1.0
        valid_ninds = np.where(valid)[0]
        # print 'Average error: ', avg_err

        # Retrieve 3D
        for tidx,pts in enumerate(self.fpts): 
            # Find valid inds that are finite
            ninds = valid_ninds[np.where(np.isfinite(pts[valid_ninds]).all(axis=1))[0]]
            xys = pts[ninds].astype(int)

            # Copy 2D data
            data.xy[ninds,tidx] = pts[ninds]

            try: 
                # Copy 3D data
                data.xyz[ninds,tidx] = Xs[tidx][xys[:,1],xys[:,0],:]

                # Copy Normals data
                if Ns is not None: 
                    data.normal[ninds,tidx] = Ns[tidx][xys[:,1],xys[:,0],:]

                ninds_ = ninds[np.isfinite(data.xyz[ninds,tidx]).all(axis=1) &
                               np.isfinite(data.normal[ninds,tidx]).all(axis=1)]
                data.idx[ninds_,tidx] = 1
            except IndexError: 
                pass

            # Mark outliers
            outliers = ninds[np.where((np.fabs(self.fpts[tidx, ninds] - 
                                               self.bpts[tidx, ninds]) > 1.0).any(axis=1))[0]]
            data.idx[outliers,tidx] = -1

        draw_utils.publish_point_cloud('tracks3d', data.xyz.reshape(-1,3), c='b')
        return data 


# import networkx as nx

# class GatedFwdBwdKLT(OfflineKLT): 
#     def __init__(self, *args, **kwargs):
#         OfflineKLT.__init__(self, *args, **kwargs)

#         self.log.debug('Initilizing GatedFwdBwdKLT')

#         # Store detected features
#         self.pts, self.fpts, self.bpts = dict(), dict(), dict()
#         self.f_flow, self.b_flow = dict(), dict()

#         # ball-tree for features (Check: memory consumption)
#         self.btindex = dict()

#         # FWD/BWD DiGraph for connectivity
#         self.g = nx.DiGraph()

#         # Process frames provided
#         self.run()

#     def node_id(self, t, n): 
#         return self.gftt_params.maxCorners * t + n

#     def node_index(self, idx): 
#         return idx / self.gftt_params.maxCorners, idx % self.gftt_params.maxCorners

#     def run(self): 
#         T = len(self.images)

#         # 1. Feature Extraction for all frames
#         st = time.time()
#         for idx,im in enumerate(self.images):
#             pts = self.process_im(im, mask=None)
#             self.pts[idx] = pts

#             # Construct tree
#             self.btindex[idx] = BallTree(pts, metric='euclidean')

#             # Note: FIX?!! fpts,bpts set
#             self.fpts[idx] = pts.copy()
#             self.bpts[idx] = pts.copy()

#         self.log.debug('Feature Extraction (N=%i): %s s' % (T, time.time() - st))

#         # 2. FWD/BWD-Flow Extraction
#         # For (1,2), (2,3), ....
#         finds = np.arange(0, T)
#         pair_inds = zip(finds[:-1], finds[1:])
#         self.log.debug('Forward flow ========> %s' % pair_inds)
#         for (idx0,idx1) in pair_inds: 
#             # FWD-flow: f_flow[1] = lk(1,2)
#             flow = self.fblk_1pass(idx0, idx1)
#             # self.f_flow[idx0] = flow


#             # a. Apply bwd predictions for valid flow regions
#             self.bpts[idx0] = np.empty_like(self.pts[idx1])
#             self.bpts[idx0].fill(np.nan)

#             bpts0 = self.pts[idx1][flow.binds] + flow.bflow[flow.binds]
#             self.bpts[idx0][flow.binds] = bpts0

#             # Find closest detected features to bwd predictions 
#             nn_inds = self.btindex[idx0].query_radius(bpts0, 1.0)
#             for src, nns in zip(flow.binds, nn_inds): 
#                 src_id = self.node_id(idx1, src)
#                 tgt_ids = [self.node_id(idx0, nn) for nn in nns]

#                 # Directed edges with 0 weight
#                 for tgt_id in tgt_ids: 
#                     self.g.add_edge(src_id, tgt_id, weight=0)


#             # b. Apply forward predictions for valid flow regions
#             self.fpts[idx1] = np.empty_like(self.pts[idx0])
#             self.fpts[idx1].fill(np.nan)

#             fpts1 = self.pts[idx0][flow.finds] + flow.fflow[flow.finds]
#             self.fpts[idx1][flow.finds] = fpts1

#             # Find closest detected features to fwd predictions 
#             nn_inds = self.btindex[idx1].query_radius(fpts1, 1.0)
#             for src, nns in zip(flow.finds, nn_inds): 
#                 src_id = self.node_id(idx0, src)
#                 tgt_ids = [self.node_id(idx1, nn) for nn in nns]

#                 # Directed edges with 0 weight
#                 for tgt_id in tgt_ids: 
#                     self.g.add_edge(src_id, tgt_id, weight=0)


#     def run_2pass(self): 
#         T = len(self.images)

#         # 1. Feature Extraction for all frames
#         st = time.time()
#         for idx,im in enumerate(self.images):
#             pts = self.process_im(im, mask=None)
#             self.pts[idx] = pts

#             # Construct tree
#             self.btindex[idx] = BallTree(pts, metric='euclidean')

#             # Note: FIX?!! fpts,bpts set
#             self.fpts[idx] = pts.copy()
#             self.bpts[idx] = pts.copy()

#         self.log.debug('Feature Extraction (N=%i): %s s' % (T, time.time() - st))

#         # 2. FWD/BWD-Flow Extraction
#         # For (1,2), (2,3), ....
#         finds = np.arange(0, T)
#         pair_inds = zip(finds[:-1], finds[1:])
#         self.log.debug('Forward flow ========> %s' % pair_inds)
#         for (idx0,idx1) in pair_inds: 
#             # FWD-flow: f_flow[1] = lk(1,2)
#             fflow0 = self.fblk(self.pts[idx0], idx0, idx1)
#             self.f_flow[idx0] = fflow0

#             # Apply forward predictions for valid flow regions
#             self.fpts[idx1] = np.empty_like(self.pts[idx0])
#             self.fpts[idx1].fill(np.nan)

#             fpts1 = self.pts[idx0][fflow0.inds] + fflow0.flow[fflow0.inds]
#             self.fpts[idx1][fflow0.inds] = fpts1

#             # Find closest detected features to fwd predictions 
#             nn_inds = self.btindex[idx1].query_radius(fpts1, 1.0)
#             for src, nns in zip(fflow0.inds, nn_inds): 
#                 src_id = self.node_id(idx0, src)
#                 tgt_ids = [self.node_id(idx1, nn) for nn in nns]

#                 # Directed edges with 0 weight
#                 for tgt_id in tgt_ids: 
#                     self.g.add_edge(src_id, tgt_id, weight=0)

#         # 3. BWD flow
#         rinds = finds[::-1]
#         pair_rinds = zip(rinds[:-1], rinds[1:])
#         self.log.debug('Backward flow ========> %s' % pair_rinds)
#         for (idx1,idx0) in pair_rinds: 
#             # BWD-flow: b_flow[2] = lk(2,1)
#             bflow1 = self.fblk(self.pts[idx1], idx1, idx0)
#             self.b_flow[idx1] = bflow1

#             # Make sure nans are not written over bpts
#             self.bpts[idx0] = np.empty_like(self.pts[idx1])
#             self.bpts[idx0].fill(np.nan)

#             bpts0 = self.pts[idx1][bflow1.inds] + bflow1.flow[bflow1.inds]
#             self.bpts[idx0][bflow1.inds] = bpts0

#             # Find closest detected features to bwd predictions 
#             nn_inds = self.btindex[idx0].query_radius(bpts0, 1.0)
#             for src, nns in zip(bflow1.inds, nn_inds): 
#                 src_id = self.node_id(idx1, src)
#                 tgt_ids = [self.node_id(idx0, nn) for nn in nns]

#                 # Directed edges with 0 weight
#                 for tgt_id in tgt_ids: 
#                     self.g.add_edge(src_id, tgt_id, weight=0)

#         self.log.debug('Nodes/Edges: %i %i' % (self.g.number_of_nodes(), self.g.number_of_edges()))


#     def fblk_1pass(self, idx0, idx1): 
#         # 1. Forward flow
#         p0 = self.pts[idx0]
#         fflow = self.fblk(p0, idx0, idx1)

#         # 2. Reverse flow
#         p1 = self.pts[idx1]
#         bflow = self.fblk(p1, idx1, idx0)

#         # Retain flow info 
#         return AttrDict({'fflow': fflow.flow, 'bflow': bflow.flow,
#                          'finds': fflow.inds, 'binds': bflow.inds})

#     # def lk_1pass(self, idx0, idx1): 
#     #     # Forward flow
#     #     p0 = self.pts[idx0]
#     #     p1hat, st1, err1 = cv2.calcOpticalFlowPyrLK(self.images[idx0], self.images[idx1], 
#     #                                            p0, None, **self.lk_params)
#     #     # Invalidate pts without good flow
#     #     inds1 = np.where(st1 == 0)[0]
#     #     p1hat[inds1,:] = np.nan

#     #     # Backward flow
#     #     p1 = self.pts[idx1]
#     #     p0hat, st0, err0 = cv2.calcOpticalFlowPyrLK(self.images[idx1], self.images[idx0], 
#     #                                            p1, None, **self.lk_params)
#     #     # Invalidate pts without good flow
#     #     inds0 = np.where(st0 == 0)[0]
#     #     p0hat[inds0,:] = np.nan

#     #     # Retain flow info 
#     #     return AttrDict({'fflow': p1hat-p0, 'bflow': p0hat-p1,
#     #                      'finds': inds1, 'binds': inds0})

#     def viz_pts(self, Xs): 
#         X = []
#         for idx,pts in self.pts.iteritems(): 

#             # Plot 
#             rgb = cv2.cvtColor(self.images[idx].copy(), cv2.COLOR_GRAY2BGR)            
#             for pt in pts: 
#                 if np.isnan(pt).any(): continue
#                 cv2.circle(rgb, tuple(map(int, pt)), 
#                            2, (0,255,0), -1, lineType=cv2.LINE_AA)

#             # Plot Trajectories in 3d
#             yxs = pts[np.where(np.isfinite(pts))[0]].astype(int)
#             try: 
#                 X.append(Xs[idx][yxs[:,1],yxs[:,0],:])
#                 # X.append(np.hstack([xs_sp, ys_sp, zs_sp]))
#             except IndexError: 
#                 pass

#             cv2.imshow('rgb', rgb)
#             cv2.waitKey(10)
#             # time.sleep(0.2)

#         draw_utils.publish_point_cloud('tracks', np.vstack(X), c='b')
