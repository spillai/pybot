import numpy as np
from pybot.geometry.rigid_transform import RigidTransform

def metric_from_gps(gps_coords, scale=None):
    assert(gps_coords.ndim == 2 and gps_coords.shape[1] == 2)
    er = 6378137.
    lat, lon = gps_coords[:,0], gps_coords[:,1]
    scale = np.cos(lat[0] * np.pi / 180.) if scale is None else 1.0
    return np.vstack([scale * lon * np.pi * er / 180.,
                      scale * er * np.log(np.tan((90. + lat) * np.pi / 360.))]).T

def bearing_from_metric_gps(mgps, look_forward=5):
    """
    Look forward by X m to identify bearing
    """
    # Use cumulative distance travelled as mechanism to
    # determine bearing
    D = np.linalg.norm(mgps[1:]-mgps[:-1], axis=1)
    Dint = np.hstack([0, np.cumsum(D)])
    
    def normalize(v):
        return v / (1e-12 + np.linalg.norm(v, axis=1)[:,np.newaxis])

    # Determine bearing
    N = len(mgps)
    vgps = []
    for idx,mgp in enumerate(mgps):
        d1 = Dint[idx]
        found = False
        for j in range(idx,N):
            if Dint[j]-d1 > look_forward:
                vgps.append(mgps[j]-mgp)
                found = True
                break
        if not found:
            vgps.append(vgps[-1])
    vgps = normalize(np.vstack(vgps))

    assert(len(mgps) == len(vgps))
    return np.arctan2(vgps[:,0], vgps[:,1])

    
