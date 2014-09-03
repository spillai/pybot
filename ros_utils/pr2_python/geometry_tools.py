#arm_planner.py
'''
A set of tools for working with ROS messages and geometry.  

Transformations are represented as Poses throughout this module, which is a dubious but convenient choice.
'''

__docformat__ = "restructuredtext en"


from pr2_python.conversions import point_to_list, list_to_point
from geometry_msgs.msg import Quaternion, Pose
import numpy as np
import tf.transformations as tt
import rospy
import copy

DIST_EPS = 0.001
'''
Default value (in cm) for considering two distances to be near.
'''

ANGLE_EPS = 0.01
'''
Default value (in radians) for considering to angles to be near.
'''


def wrap_angle(a):
    '''
    Wrap an angle to [0, 2\pi].

    **Args:**

        **a (double):** angle

    **Returns:**
        The angle equivalent to a between 0 and 2\pi
    '''
    return (a % (2.0*np.pi))

def angular_distance(a1, a2):
    '''
    Returns the shortest distance in radians between to angles.

    **Args:**

        **a1, a2 (double):** angles

    **Returns:**
        The shortest distance in radians between a1 and a2
    '''
    a1 = wrap_angle(a1)
    a2 = wrap_angle(a2)
    da = wrap_angle(a1 - a2)
    if da > np.pi:
        da = wrap_angle(a2 - a1)
    return da
    

#distance between poses and quaternions
def euclidean_distance(p1, p2):
    '''
    Returns the Euclidean distance between two points.

    **Args:**

        **p1, p2 (geometry_msgs.msg.Point):** points

    **Returns:**
        The Euclidean distance between p1 and p2 in three dimensions.
    '''
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dz = p2.z - p1.z
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def euclidean_distance_xy(p1, p2):
    '''
    Returns the Euclidean distance in the xy plane between two points.

    **Args:**

        **p1, p2 (geometry_msgs.msg.Point):** points

    **Returns:**
        The Euclidean distance between p1 and p2 in the xy plane.
    '''
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return np.sqrt(dx*dx + dy*dy)


def near(pose1, pose2, dist_eps=DIST_EPS, angle_eps=ANGLE_EPS):
    '''
    Checks if two poses are near each other.

    **Args:**

        **pose1, pose2 (geometry_msgs.msg.Pose):** poses
        
        **dist_eps (double):** The acceptable discrepancy (in m) in Euclidean space
        
        **angle_eps (double):** The acceptable discrepancy (in radians) in angular space
    
    **Returns:**
        True if the euclidean distance between pose1 and pose2 is less than dist_eps and the angular distance is
        less than angle_eps.
    '''
    return euclidean_distance(pose1.position, pose2.position) < dist_eps and\
        quaternion_distance(pose1.orientation, pose2.orientation) < angle_eps

#working with quaternions
def quaternion_distance(q1, q2):
    '''
    Returns the distance between two quaternions.

    **Args:**

        **q1, q2 (geometry_msgs.msg.Quaternion):** quaternions

    **Returns:**
        The angular distance between q1 and q2.
    '''
    inprod = q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w
    ret = np.arccos(2.0*inprod - 1.0)
    if np.isnan(ret):
        return 0.0
    return ret

def rotation_matrix(quat):
    '''
    Creates a rotation matrix from a quaternion

    **Args:**

        **quat (geometry_msgs.msg.Quaternion):** quaternion

    **Returns:**
        A list of lists representing the rotation matrix corresponding to this quaternion
    '''
    mat = []
    row0 = [1.0 - 2*quat.y*quat.y - 2*quat.z*quat.z,
            2*quat.x*quat.y - 2*quat.z*quat.w,
            2*quat.x*quat.z + 2*quat.y*quat.w]
    mat.append(row0)
    row1 = [2*quat.x*quat.y + 2*quat.z*quat.w,
            1 - 2*quat.x*quat.x - 2*quat.z*quat.z,
            2*quat.y*quat.z - 2*quat.x*quat.w]
    mat.append(row1)
    row2 = [2*quat.x*quat.z - 2*quat.y*quat.w,
            2*quat.y*quat.z + 2*quat.x*quat.w,
            1 - 2*quat.x*quat.x - 2*quat.y*quat.y]
    mat.append(row2)
    return mat


def euler_to_quaternion(phi, theta, psi):
    '''
    Converts Euler angles to a quaternion

    **Args:**

        **phi, theta, psi (double):** Euler angles

    **Returns:**
        The geometry_msgs.msg.Quaternion corresponding to the Euler angles
    '''
    q = Quaternion()
    q.x = np.sin(phi/2.0)*np.cos(theta/2.0)*np.cos(psi/2.0) -\
        np.cos(phi/2.0)*np.sin(theta/2.0)*np.sin(psi/2.0)
    q.y = np.cos(phi/2.0)*np.sin(theta/2.0)*np.cos(psi/2.0) +\
        np.sin(phi/2.0)*np.cos(theta/2.0)*np.sin(psi/2.0)
    q.z = np.cos(phi/2.0)*np.cos(theta/2.0)*np.sin(psi/2.0) -\
        np.sin(phi/2.0)*np.sin(theta/2.0)*np.cos(psi/2.0)
    q.w = np.cos(phi/2.0)*np.cos(theta/2.0)*np.cos(psi/2.0) +\
        np.sin(phi/2.0)*np.sin(theta/2.0)*np.sin(psi/2.0)
    return q

def quaternion_to_euler(quat):
    '''
    Converts a quaternion to Euler angles

    **Args:**

        **quat (geometry_msgs.msg.Quaternion):** quaternion

    **Returns:**
        (phi, theta, psi), the euler angles corresponding to quat
    '''
    return tt.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

def multiply_quaternions(q1, q2):
    '''
    Multiplies two quaternions

    **Args:**

        **q1, q2 (geometry_msgs.msg.Quaternion):** quaternions

    **Returns:**
        A geometry_msgs.msg.Quaternion that equals q1*q2
    '''
    q = Quaternion()
    q.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    q.x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
    q.y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
    q.z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
    return q

def invert_quaternion(q):
    '''
    Inverts a quaternion

    **Args:**

        **q (geometry_msgs.msg.Quaternion):** quaternion

    **Returns:**
        A geometry_msgs.msg.Quaternion that is the inverse of q
    '''
    qt = copy.copy(q)
    qt.w *= -1.0
    return qt

#transforming using poses as transforms

def transform_point(point, pose):
    '''
    Transforms a point using pose as the transformation.

    Takes a point defined in the frame defined by pose (i.e. the frame in which pose is the origin) and returns
    it in the frame in which pose is defined.  Calling this with the point (0, 0, 0) will return pose.  This
    is useful, for example, in finding the corner of a box arbitrarily positioned in space.  In the box's frame
    the corner is (xdim, ydim, zdim).  In the world, the corner is at transform_point(corner, box_pose).
    
    **Args:**
    
        **point (geometry_msgs.msg.Point):** Point to transform

        **pose (geometry_msgs.msg.Pose):** The transform
    
    **Returns:**
        A geometry_msgs.msg.Point
    '''
    return list_to_point(transform_list(point_to_list(point), pose))

def transform_list(point, pose):
    '''
    Transforms a list using pose as the transform

    Takes a point defined in the frame defined by pose (i.e. the frame in which pose is the origin) and returns
    it in the frame in which pose is defined.  Calling this with the point (0, 0, 0) will return pose.  This
    is useful, for example, in finding the corner of a box arbitrarily positioned in space.  In the box's frame
    the corner is (xdim, ydim, zdim).  In the world, the corner is at transform_point(corner, box_pose).
    
    **Args:**

        **point ((double, double, double)):** Point to transform as (x, y, z)

        **pose (geometry_msgs.msg.Pose):** The transform
    
    Returns:
        (x', y', z')
    '''
    translation = point_to_list(pose.position)
    rotMatrix = rotation_matrix(pose.orientation)

    newpoint = []
    for row in range(0,3):
        newpoint.append(rotMatrix[row][0]*point[0] + 
                        rotMatrix[row][1]*point[1] +
                        rotMatrix[row][2]*point[2] + 
                        translation[row])
    return newpoint

def inverse_transform_point(point, pose):
    '''
    Inverse transforms a point using pose as the transform

    Takes a point defined in the frame defined in which pose is defined and converts it into the frame defined
    by pose (i.e. the frame in which pose is the origin).  This is useful, for example, if we know the position
    of the object in the world is (x, y, z) and the pose of a table is table_pose.  We can find the height of
    the object above the table by transforming it into the pose of the table using 
    transform_point(point, table_pose) and looking at the z coordinate of the result.
    
    **Args:**

        **point (geometry_msgs.msg.Point):** Point to transform

        **pose (geometry_msgs.msg.Pose):** The transform
    
    **Returns:**
        A geometry_msgs.msg.Point
    '''
    return list_to_point(inverse_transform_list(point_to_list(point), pose))

def inverse_transform_list(point, pose):
    '''
    Inverse transforms a list using pose as the transform.

    Takes a point defined in the frame defined in which pose is defined and converts it into the frame defined
    by pose (i.e. the frame in which pose is the origin).  This is useful, for example, if we know the position
    of the object in the world is (x, y, z) and the pose of a table is table_pose.  We can find the height of
    the object above the table by transforming it into the pose of the table using 
    transform_point(point, table_pose) and looking at the z coordinate of the result.
    
    **Args:**

        **point ((double, double, double)):** Point to transform as (x, y, z)

        **pose (geometry_msgs.msg.Pose):** The transform

    **Returns:**
        (x', y', z')
    '''
    translation = point_to_list(pose.position)
    rotMatrix = rotation_matrix(pose.orientation)

    newpoint = []
    for row in range(0,3):
        newpoint.append(rotMatrix[0][row]*(point[0]-translation[0]) +
                        rotMatrix[1][row]*(point[1]-translation[1]) +
                        rotMatrix[2][row]*(point[2]-translation[2]))
    return newpoint

def transform_quaternion(quaternion, transform):
    '''
    Transforms a quaternion

    **Args:**

        **quaternion (geometry_msgs.msg.Quaternion):** quaternion to transform
        
        **transform (geometry_msgs.msg.Pose):** transform

    **Returns:**
        transform.orientation*quaternion as geometry_msgs.msg.Quaternion
    '''
    return multiply_quaternions(transform.orientation, quaternion)

def inverse_transform_quaternion(quaternion, transform):
    '''
    Inverse transforms a quaternion

    **Args:**

        **quaternion (geometry_msgs.msg.Quaternion):** quaternion

        **transform (geometry_msgs.msg.Pose):** transform

    **Returns:**
        (transform.orientation)^{-1}*quaternion as geometry_msgs.msg.Quaternion
    '''
    return multiply_quaternions(invert_quaternion(transform.orientation), quaternion)

def transform_pose(pose_in, transform):
    '''
    Transforms a pose

    Takes a pose defined in the frame defined by transform (i.e. the frame in which transform is the origin) and 
    returns it in the frame in which transform is defined.  Calling this with the origin pose will return transform.
    This returns
        
        pose.point = transform_point(pose_in.point, transform)
        
        pose.orientation = transform_orientation(pose_in.orientation, transform)
    
    **Args:**

        **pose (geometry_msgs.msg.Pose):** Pose to transform

        **transform (geometry_msgs.msg.Pose):** The transform
    
    **Returns:**
        A geometry_msgs.msg.Pose
    '''
    pose = Pose()
    pose.position = transform_point(pose_in.position, transform)
    pose.orientation = transform_quaternion(pose_in.orientation, transform)
    return pose

def inverse_transform_pose(pose_in, transform):
    '''
    Inverse transforms a pose

    Takes a pose defined in the frame defined in which transform is defined and converts it into the frame defined
    by transform (i.e. the frame in which transform is the origin).  This returns
        
        pose.point = inverse_transform_point(pose_in.point, transform)
        
        pose.orientation = inverse_transform_orientation(pose_in.orientation, transform)
    
    **Args:**

        **pose (geometry_msgs.msg.Pose):** Pose to transform
        
        **transform (geometry_msgs.msg.Pose):** The transform
    
    **Returns:**
        A geometry_msgs.msg.Pose
    '''
    pose = Pose()
    pose.position = inverse_transform_point(pose_in.position, transform)
    pose.orientation = inverse_transform_quaternion(pose_in.orientation, transform)
    return pose

def bounding_box_corners(shape):

    dimensions = shape.dimensions
    if shape.type == shape.BOX:
        minx = -dimensions[0]/2.0
        maxx = dimensions[0]/2.0
        miny = -dimensions[1]/2.0
        maxy = dimensions[1]/2.0
        minz = -dimensions[2]/2.0
        maxz = dimensions[2]/2.0
    elif shape.type == shape.CYLINDER:
        minx = -dimensions[1]
        maxx = dimensions[1]
        miny = minx
        maxy = maxx
        minz = -dimensions[2]/2.0
        maxz = dimensions[2]/2.0
    elif shape.type == shape.SPHERE:
        minx = -dimensions[0]
        maxx = dimensions[0]
        miny = minx
        maxy = maxx
        minz = minx
        maxz = maxx
    elif shape.type == shape.MESH:
        minx = float('inf')
        maxx = -float('inf')
        miny = float('inf')
        maxy = -float('inf')
        minz = float('inf')
        maxz = -float('inf')
        for p in shape.vertices:
            if p.x < minx:
                minx = p.x
            if p.x > maxx:
                maxx = p.x
            if p.y < miny:
                miny = p.y
            if p.y > maxy:
                maxy = p.y
            if p.z < minz:
                minz = p.z
            if p.z > maxz:
                maxz = p.z
    return [[minx, miny, minz], [minx, miny, maxz], [minx, maxy, minz], [minx, maxy, maxz],
            [maxx, miny, minz], [maxx, miny, maxz], [maxx, maxy, minz], [maxx, maxy, maxz]]
                

def bounding_box_dimensions(shape):
    '''
    Returns the bounding box of a single shape.

    **Args:**
        **shape (arm_navigation_msgs.msg.Shape): shape
     
    **Returns:**
        The dimensions of the bounding box as (x, y, z)
    '''
    corners = bounding_box_corners(shape)
    return (corners[-1][0]-corners[0][0], corners[-1][1] - corners[0][1], corners[-1][2] - corners[0][2])
