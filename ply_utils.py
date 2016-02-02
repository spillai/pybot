import os, cv2
import numpy as np

global ply_header
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    global ply_header
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


# Write ply for testing purposes
def write_ply_with_dispmask(fn, left_im, disp, X): 
    mask = disp > disp.min()
    colors = cv2.cvtColor(left_im, cv2.COLOR_GRAY2RGB)
    write_ply(fn, X[mask], colors[mask])
    print '%s saved' % fn

