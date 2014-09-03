#!/usr/bin/env python 
#find_free_space.py
'''
Functions for finding free space on a table.
'''

from __future__ import division

__docformat__ = "restructuredtext en"


import roslib
roslib.load_manifest('pr2_python')
import rospy
import random
import pr2_python.geometry_tools as gt
import pr2_python.transform_listener as tl
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped
from pr2_python.point_on_table import point_on_table
import copy
import pr2_python.visualization_tools as vt
from visualization_msgs.msg import MarkerArray, Marker


NUM_TRIES = 100

############################################################
# External API
############################################################

vpub = rospy.Publisher('free_space_markers', MarkerArray)
marray = MarkerArray()

def free_spots_on_table(table, obj, orientation, blocking_objs, res=0.1):

    object_pose = Pose()
    object_pose.orientation = orientation

    #these are the corners of the object in the object's frame
    object_frame_corners = gt.bounding_box_corners(obj.shapes[0])
    #rospy.loginfo('Corners are')
    #for c in object_frame_corners: rospy.loginfo(str(c))
    #these are the corners after we apply the rotation
    header_frame_corners = [gt.transform_list(c, object_pose) for c in object_frame_corners]
    #these are the corners relative to the table's lower left corner
    table_frame_corners = [gt.inverse_transform_list(c, table.poses[0]) for c in header_frame_corners]
    #the footprint of the object above the table
    oxmax = max([c[0] for c in table_frame_corners])
    oxmin = min([c[0] for c in table_frame_corners])
    oymax = max([c[1] for c in table_frame_corners])
    oymin = min([c[1] for c in table_frame_corners])
    object_dims = (oxmax - oxmin, oymax - oymin)
    rospy.loginfo('Object dimensions are '+str(object_dims))
    obj_poses = free_spots_from_dimensions(table, object_dims, blocking_objs, res=res)
    #return these as proper object poses with the object sitting on the table
    #the height of the object in the world in its place orientation
    obj_height = -1.0*min([c[2] for c in header_frame_corners])
    above_table = gt.transform_list([0, 0, obj_height], table.poses[0])
    place_height = above_table[2]
    for pose in obj_poses:
        pose.pose.position.z = place_height
        pose.pose.orientation = copy.copy(orientation)
    return obj_poses

def free_spots_from_dimensions(table, object_dims, blocking_objs, res=0.1):
    '''
    Currently works only with a single shape

    Assumes table is in xy plane
    '''
    #find the lower left corner of the table in the header frame
    #we want everything relative to this point
    table_corners = gt.bounding_box_corners(table.shapes[0])
    lower_left = Pose()
    lower_left.position = gt.list_to_point(table_corners[0])
    lower_left.orientation.w = 1.0
    #rospy.loginfo('Table corners are:')
    #for c in table_corners: rospy.loginfo('\n'+str(c))
    #rospy.loginfo('Lower left =\n'+str(lower_left))
    #rospy.loginfo('Table header =\n'+str(table.header))
    #rospy.loginfo('Table pose =\n'+str(table.poses[0]))
    #this is the position of the minimum x, minimum y point in the table's header frame
    table_box_origin = gt.transform_pose(lower_left, table.poses[0])
    tr = gt.inverse_transform_list(gt.transform_list(table_corners[-1], table.poses[0]), table_box_origin)
    table_dims = (tr[0], tr[1])
    tbos = PoseStamped()
    tbos.header = table.header
    tbos.pose = table_box_origin
    marray.markers.append(vt.marker_at(tbos, ns='table_origin', mtype=Marker.CUBE, r=1.0))
    max_box = PointStamped()
    max_box.header = table.header
    max_box.point = gt.list_to_point(gt.transform_list([table_dims[0], table_dims[1], 0], table_box_origin)) 
    marray.markers.append(vt.marker_at_point(max_box, ns='table_max', mtype=Marker.CUBE, r=1.0))

    #rospy.loginfo('Table box origin is '+str(table_box_origin)+' dimensions are '+str(table_dims))

                
    locs_on_table =  _get_feasible_locs(table_dims, object_dims, res)
    for i, l in enumerate(locs_on_table):
        pt = Point()
        pt.x = l[0]
        pt.y = l[1]
        mpt = PointStamped()
        mpt.header = table.header
        mpt.point = gt.transform_point(pt, table_box_origin)
        marray.markers.append(vt.marker_at_point(mpt, mid=i, ns='locations', r=1.0, g=1.0, b=0.0))

    feasible_locs = []
    #check that these points really are on the table
    for i, l in enumerate(locs_on_table):
        pt = Point()
        pt.x = l[0]
        pt.y = l[1]
        #this point is now defined relative to the origin of the table (rather than its minimum x, minimum y point)
        table_pt = gt.inverse_transform_point(gt.transform_point(pt, table_box_origin), table.poses[0])
        if point_on_table(table_pt, table.shapes[0]):
            feasible_locs.append(l)
            marray.markers[i+2].color.r = 0.0
            marray.markers[i+2].color.b = 1.0
    rospy.loginfo('Testing '+str(len(feasible_locs))+' locations')
    if not feasible_locs:
        return feasible_locs

    forbidden=[]
    for i, o in enumerate(blocking_objs):
        ofcs = gt.bounding_box_corners(o.shapes[0])
        objpose = tl.transform_pose(table.header.frame_id, o.header.frame_id, o.poses[0])
        hfcs = [gt.transform_list(c, objpose) for c in ofcs]
        tfcs = [gt.inverse_transform_list(c, table_box_origin) for c in hfcs]
        oxmax = max([c[0] for c in tfcs])
        oxmin = min([c[0] for c in tfcs])
        oymax = max([c[1] for c in tfcs])
        oymin = min([c[1] for c in tfcs])
        forbidden.append(((oxmin, oymin), (oxmax - oxmin, oymax - oymin)))
        #rospy.loginfo('\n'+str(forbidden[-1]))
        ps = PoseStamped()
        ps.header = table.header
        ps.pose = objpose
        ps = PoseStamped()
        ps.header = table.header
        ps.pose.position = gt.list_to_point(gt.transform_list([oxmin, oymin, 0], table_box_origin))
        ps.pose.orientation.w = 1.0
        marray.markers.append(vt.marker_at(ps, ns='forbidden', mid=i, r=1.0, b=0.0))
        

    # Remove forbidden rectangles
    for (bottom_left, dims) in forbidden:
        _remove_feasible_locs(feasible_locs, object_dims,
                              bottom_left,
                              _add(bottom_left, dims),
                              res)
    rospy.loginfo('There are '+str(len(feasible_locs))+' possible locations')
    obj_poses = []
    for i, fl in enumerate(feasible_locs):
        table_frame_pose = Pose()
        table_frame_pose.position.x = fl[0] + object_dims[0]/2.0
        table_frame_pose.position.y = fl[1] + object_dims[1]/2.0
        table_frame_pose.orientation.w = 1.0
        pose = PoseStamped()
        pose.header = copy.deepcopy(table.header)
        pose.pose = gt.transform_pose(table_frame_pose, table_box_origin)
        obj_poses.append(pose)
        pt = PointStamped()
        pt.header = table.header
        pt.point = pose.pose.position
        marray.markers.append(vt.marker_at_point(pt, mid=i, ns='final_locations', g=1.0, b=0.0))

    #rospy.loginfo('Object poses are:')
    #for op in obj_poses: rospy.loginfo(str(op))
    for i in range(10):
        vpub.publish(marray)
        rospy.sleep(0.1)
    return obj_poses

def free_spots(table, objects, res=0.1, forbidden=[]):
    """
    Find free spots on a table

    @param table: Dimensions of table
    @type table: Tuple (x, y)
    @param objects: Object dimensions
    @type objects: List of tuples of form (x, y)
    @param res: Resolution
    @type res: Positive float 
    @param forbidden: Forbidden rectangles
    @type forbidden: List of tuples ((x0, y0), (x_size, y_size)) where (x0, y0) is the bottom left.
    @return: list of positions of (lower left corners of) objects, or None
    """
    feasible_locs = {}
    for i, o in enumerate(objects):
        feasible_locs[i] = _get_feasible_locs(table, o, res)
    locs = {}

    # Remove forbidden rectangles
    for (bottom_left, dims) in forbidden:
        for i, o in enumerate(objects):
            _remove_feasible_locs(feasible_locs[i], o,
                                 bottom_left,
                                 _add(bottom_left, dims),
                                 res)
                                 
    # Place the objects one by one
    for i, o in enumerate(objects):
        if len(feasible_locs[i])==0:
            rospy.logdebug("No feasible locs for object {0}".format(i))
            return None
        locs[i] = random.sample(feasible_locs[i], 1)[0]
        rospy.logdebug("Trying {0} for object {1}".format(locs[i], i))
        for j in range(i+1, len(objects)):
            _remove_feasible_locs(feasible_locs[j], objects[j], locs[i],
                                 _add(locs[i], objects[i]), res)
    return locs


############################################################
# Internal
############################################################

def _get_feasible_locs(table, o, res):
    locs = set()
    rx = int((table[0]-o[0])/res)+1
    for x in range(rx):
        for y in range(int((table[1]-o[1])/res)+1):
             locs.add((x*res,y*res))
    return locs

def _remove_feasible_locs(locs, o, bottom_left, top_right, res):

    xmin = bottom_left[0] - o[0]
    xmax = top_right[0]
    ymin = bottom_left[1] - o[1]
    ymax = top_right[1]
    remove = []
    for l in locs:
        if l[0] > xmin and l[0] < xmax and l[1] > ymin and l[1] < ymax:
            remove.append(l)
    for r in remove:
        locs.remove(r)

def _add(v1, v2):
    return [x+y for (x,y) in zip(v1, v2)]

def _tuplize(vec2d):
    return (vec2d.x, vec2d.y)
