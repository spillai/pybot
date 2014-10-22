import cv2, os, logging
import numpy as np
import bot_externals.draw_utils as draw_utils

# Class to encapsulate all relevant data ===============================
# Build data (idx id, idx utime -> idx feature) 
class Feature3DData: 
    def __init__(self, feature_data=None, discretize_ms=30): 
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('===> Loading Feature3DData ...')
        if feature_data is None: return 

        # Skips
        utime_skip = max(1,int(discretize_ms * 1. / 30))

	# Enumerate data, and build utime/id index
	self.feature_ids = np.unique([ feat['id'] for feat in feature_data ])
	self.feature_utimes = np.unique([ feat['utime'] for feat in feature_data ])
        self.feature_utimes = self.feature_utimes[::utime_skip]

	self.num_feats, self.num_utimes = len(self.feature_ids), len(self.feature_utimes)
	self.log.debug('Num features: %i' % self.num_feats)
	self.log.debug('Num utimes: %i' % self.num_utimes)

	self.ids2inds = dict(zip(self.feature_ids, np.arange(self.num_feats)))
	self.utimes2inds = dict(zip(self.feature_utimes, np.arange(self.num_utimes)))

	# Store indices (idx id, idx utime -> idx feature)
	self.idx = np.zeros((self.num_feats, self.num_utimes), dtype=np.int32)
	self.idx.fill(-1)

	# Store xyz, normals, tangents
	self.xy = np.zeros((self.num_feats, self.num_utimes, 2), dtype=np.float32)
	self.xyz = np.zeros((self.num_feats, self.num_utimes, 3), dtype=np.float32)
	self.normal = np.zeros((self.num_feats, self.num_utimes, 3), dtype=np.float32)

        # Main data container
	for idx,feat in enumerate(feature_data): 
	    if not feat['utime'] in self.utimes2inds: continue
	    i,j = self.ids2inds[feat['id']], self.utimes2inds[feat['utime']]
	    self.idx[i,j]  = idx
	    self.xy[i,j] = feat['point'].reshape(-1)
	    self.xyz[i,j] = feat['xyz'].reshape(-1)
	    self.normal[i,j] = feat['normal'].reshape(-1) 
            self.normal[i,j] *= 1.0 / np.linalg.norm(self.normal[i,j])
            # print self.xyz[i,j]

            if np.isnan(self.xyz[i,j]).any() or np.isnan(self.normal[i,j]).any() \
               or np.fabs(np.linalg.norm(self.normal[i,j])-1) > 1e-2: 
                self.idx[i,j] = -1

        # Save valid_inds for future pruning
        self.valid_feat_inds = np.arange(self.num_feats)

        # Checks
        all_inds = np.where(self.idx != -1)
        assert( not np.isnan(self.xyz[all_inds]).any() )
        assert( not np.isnan(self.normal[all_inds]).any() )

    @staticmethod
    def empty(N, T): 
        data = Feature3DData()
        data.feature_ids = np.arange(N)
        data.feature_utimes = np.arange(T)

        data.num_feats, data.num_utimes = N, T
        data.ids2inds = dict(zip(data.feature_ids, np.arange(N)))
        data.utimes2inds = dict(zip(data.feature_utimes, np.arange(T)))

	# Store indices (idx id, idx utime -> idx feature)
	data.idx = np.zeros((data.num_feats, data.num_utimes), dtype=np.int32)
	data.idx.fill(-1)

	# Store xyz, normals, tangents
	data.xy = np.empty((data.num_feats, data.num_utimes, 2), dtype=np.float32)
	data.xyz = np.empty((data.num_feats, data.num_utimes, 3), dtype=np.float32)
	data.normal = np.empty((data.num_feats, data.num_utimes, 3), dtype=np.float32)

        # Initialize to nans
        data.xy.fill(np.nan), data.xyz.fill(np.nan), data.normal.fill(np.nan)

        # Save valid_inds for future pruning
        data.valid_feat_inds = np.arange(data.num_feats)
        return data


    def prune_by_length(self, min_track_length=0, verbose=True):
        """
        Prune tracks based on length and return indices of valid tracks
        """
        # Sum over all UTIMES, and check nz for x value, and nz != 1
        self.log.debug('===> Pruning features by length ...')
        valid_utimes_count = np.sum(self.idx[:,:] != -1, axis=1)
        self.valid_feat_inds, = np.where(valid_utimes_count > min_track_length)
        if verbose: 
            self.log.debug('--- Done pruning missing features: %i out of %i good' % \
                (len(self.valid_feat_inds), self.num_feats))
        return 

    def pick_top_lengths(self, k=100, verbose=True):
        """
        Pick top tracks based on length
        """
        self.log.debug('===> Picking top features by length ...')
        valid_utimes_count = np.sum(self.idx[:,:] != -1, axis=1)
        sorted_inds = sorted(zip(range(len(valid_utimes_count)), valid_utimes_count), 
                                     key=lambda x: x[1], reverse=True)
        top_inds, _ = zip(*sorted_inds)
        self.valid_feat_inds = np.array(top_inds)[:k]
        self.log.debug('%s' % self.valid_feat_inds)
        # if verbose: 
        self.log.debug('--- Done picking missing features: %i out of %i good' % \
            (len(self.valid_feat_inds), self.num_feats))
        return 

    def pick_top_by_displacement(self, k=20, verbose=True):
        """
        Pick top tracks based on total distance covered by track
        """
        self.log.debug('===> Pruning features by displacement ...')
        top_inds = []
        for ind in self.valid_feat_inds: 
            ut_inds, = np.where(self.idx[ind,:] != -1)
            if not len(ut_inds): continue
            X = self.xy[ind,ut_inds].reshape((-1,2))
            Xmin, Xmax = np.min(X, axis=0), np.max(X, axis=0);
            top_inds.append((ind, np.linalg.norm(np.fabs(Xmin-Xmax)) ))

        # Reverse true: for longest traj, false: for shortest traj
        top_inds.sort(key=lambda x: x[1], reverse=True)
        self.valid_feat_inds = np.array([ ind for ind,score in top_inds[:k] ])
        self.log.debug('Done picking top by displacement: %i good' % (len(self.valid_feat_inds)))
        
        # Setting the rest to invalid
        for ind,score in top_inds[k:]: 
            ut_inds, = np.where(self.idx[ind,:] != -1)
            self.idx[ind, ut_inds] = -1;
        return 

    def prune_discontinuous(self, min_feature_continuity_distance=0, 
                            min_normal_continuity_angle=1.57, verbose=True):
        """
        Prune features in valid tracks that are not continuous (in xyz)
        """
        self.log.debug('===> Prune features that are not continuous ...')
        pruned = 0
        for ind in self.valid_feat_inds: 
            ut_inds, = np.where(self.idx[ind,:] != -1)
            ut_inds_p, ut_inds_n = ut_inds[:-1], ut_inds[1:]
            invalid_ut_inds, = np.where(
                np.bitwise_or(np.linalg.norm(self.xyz[ind,ut_inds_p]-
                                             self.xyz[ind,ut_inds_n], 
                                             axis=1) > min_feature_continuity_distance, 
                              np.sum(self.normal[ind,ut_inds_p] * 
                                     self.normal[ind,ut_inds_n], 
                                     axis=1) < min_normal_continuity_angle))
            
            # Invalidate entire trakc if any discontinuity
            if len(invalid_ut_inds): 
                self.idx[ind,:] = -1
                pruned += 1
        self.log.debug('Erasing %i out of %i ' % (pruned, len(self.valid_feat_inds)))
        return 

    # Viz data =============================================================
    def viz_data(self, utimes_inds=None): 
        if utimes_inds is None: utimes_inds = np.arange(self.num_utimes)

        # Draw pruned features =================================================
        viz_pts, viz_normals, viz_idloc, viz_text = [], [], [], []
        viz_traj1, viz_traj2 = [], []
        for idx in self.valid_feat_inds:

            # Valid utimes
            ut_inds, = np.where(self.idx[idx,utimes_inds] != -1)
            ut_inds = utimes_inds[ut_inds]

            # viz_idloc.append(self.xyz[idx,ut_inds[-1],:])
            viz_text.append(str(self.feature_ids[idx]))

            # self.log.debug(feature_ids[idx], ((feature_ids[idx]) % len(feature_ids))*1.0 / len(feature_ids)

            viz_pts.append(self.xyz[idx,ut_inds,:])
            
            viz_traj1.append(self.xyz[idx,ut_inds[:-1],:])
            viz_traj2.append(self.xyz[idx,ut_inds[1:],:])

            viz_normals.append(self.xyz[idx,ut_inds,:] + self.normal[idx,ut_inds,:]*0.04)

        if not len(viz_pts): return

        viz_pts = np.vstack(viz_pts)
        viz_normals = np.vstack(viz_normals)
        viz_traj1, viz_traj2 = np.vstack(viz_traj1), np.vstack(viz_traj2)
        # viz_idloc = np.vstack(viz_idloc)

        print 'Publishing', len(viz_pts)
        draw_utils.publish_cloud('PRUNED_PTS', viz_pts, c='b', frame_id='KINECT')
        draw_utils.publish_line_segments('PRUNED_TRAJ', viz_traj1, viz_traj2, c='b', frame_id='KINECT')

        # draw_utils.publish_line_segments('PRUNED_NORMAL', viz_pts, viz_normals, c='b', frame_id='KINECT')
        # draw_utils.publish_text_lcmgl('IDS', viz_ids)
        # draw_utils.publish_text_list('IDS', viz_idloc, viz_text);

    # Determine Residual error given ground truth class that compute motion err 
    def compute_residual(self, gt_data, finds):

        agg_track_error = []
        error_info = namedtuple('error_info', ['residual', 'duration'])

        for idx in finds: 
            ut_inds, = np.where(self.idx[idx,:] != -1)

            utime_init_ind = ut_inds[0]
            utime_init = self.feature_utimes[utime_init_ind]

            finit = self.xy[idx,utime_init_ind]
            Xinit = self.xyz[idx,utime_init_ind]

            # [1:]: all but first, [-1:] last
            utime_rest_ind = ut_inds[1::50]
            nactual = len(utime_rest_ind)
            factual = self.xy[idx,utime_rest_ind]
            Xactual = self.xyz[idx,utime_rest_ind]

            for ut_ind, fa, Xa in zip(utime_rest_ind, factual, Xactual): 
                err = gt_data.motion_err(utime_init, self.feature_utimes[ut_ind], finit, fa, Xinit, Xa)

                # If None, pose unavailable
                if err is None: continue
                agg_track_error.append(error_info(residual=err, 
                                                  duration=self.feature_utimes[ut_ind]-utime_init))

            # Aggregated tracks Error
            # agg_track_error.extend(track_error)
        return agg_track_error

    def write_video(self, frames, filename): 
        # Get config params
        writer = cv2.VideoWriter(filename, 
                                 cv2.VideoWriter_fourcc('m','p','4','2'), 
                                 15.0, (640, 480), True)

        # For tracks
        tracks = defaultdict(lambda: deque(maxlen=20))
        tfeat = namedtuple('feat', ['pt','label','id'])

        # Viz
        radius, thickness = 2, 2
        clabel = dict(
            [(label, 
              tuple((draw_utils.get_color_arr(label, 1, color_by='label', flip_rb=True) * 255)
                    .astype(np.int32).flatten().tolist()))
             for label in np.unique(self.labels)])
            
        # Construct video with all frames
        for ut_idx, frame in enumerate(frames):
            img = frame.getRGB()

            # Plot all points for the corresponding utime
            detected = set()
            for find in self.valid_feat_inds: 
                if not np.all(self.xy[find,ut_idx] > 0): 
                    continue
                tracks[self.feature_ids[find]].append(
                    tfeat(label=self.labels[find], pt=self.xy[find,ut_idx].astype(int), 
                          id=self.feature_ids[find]))
                detected.add(self.feature_ids[find])

            # Draw tracks
            for tid, track in tracks.iteritems(): 
                if tid not in detected: continue
                if track[-1].label < 0: continue
                carr = clabel[track[-1].label]
                
                # Plot keypoints
                cv2.circle(img, tuple(track[-1].pt.reshape(-1).astype(int).tolist()), 
                           radius, carr, 
                           thickness, lineType=cv2.LINE_AA);

                # Plot track
                pts = [np.int32(fpt.pt.reshape(-1)) for fpt in track]
                cv2.polylines(img,[np.array(pts, np.int0)], False, 
                              carr, lineType=cv2.LINE_AA, 
                              thickness=thickness)

            # Print debug info
            if ut_idx % 50 == 0: 
                self.log.debug('Processing Frame: %i' % ut_idx)
            writer.write(img)
        
        writer.release()
        self.log.debug('--- Done writing frames')
