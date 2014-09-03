#point_on_table.py
'''
Functions for determining if a point is on a table defined by a mesh.
'''

__docformat__ = "restructuredtext en"

import numpy as np

def same_side(p1, p2, a, b):
    cp1 = np.cross(b-a, p1-a)
    cp2 = np.cross(b-a, p2-a)
    if np.dot(cp1, cp2) >= 0:
        return True
    else:
        return False

def point_in_triangle(p, v_i, v_j, v_k):
    '''
    Check whether a 2d point is inside of a triangle.

    Borrowed from http://www.blackpawn.com/texts/pointinpoly/default.html

    **Args:**

        **p (np.array):** 2d point to check.
        
        **v_i, v_j, v_k (np.array):** Vertices of triangle.

    **Returns:**
        (boolean): True if point is inside triangle, False otherwise.
    '''
    if (same_side(p, v_i, v_j, v_k) and
        same_side(p, v_j, v_k, v_i) and
        same_side(p, v_k, v_i, v_j)):
        return True
    else:
        return False

def point_on_table(point, table):
    '''
    Check whether a point is over a table.

    Approximates table as a convex 2d shape. Makes the simplifying assumption
    that the table is always in the X-Y plane.

    **Args:**

        **point (geometry_msgs.msg.Point):** Point to check.

        **table (arm_navigation_msgs.msg.Shape):** Table to check. Uses vertices.


    **Returns:**
        True if point is over the convex hull of the table.
    '''
    p = np.array([point.x, point.y])
    vertices = [np.array((v.x, v.y)) for v in table.vertices]
    for tri_i in range(0, len(table.triangles), 3):
        i = table.triangles[tri_i]
        j = table.triangles[tri_i+1]
        k = table.triangles[tri_i+2]
        v_i = vertices[i]
        v_j = vertices[j]
        v_k = vertices[k]
        if point_in_triangle(p, v_i, v_j, v_k):
            return True
    return False
        
    
if __name__ == '__main__':
    import roslib; roslib.load_manifest('simple_utils')
    from matplotlib import pyplot as plt
    from geometry_msgs.msg import Point
    from arm_navigation_msgs.msg import Shape

    table = Shape()
    
    table.vertices = [
        Point(0.0, 0.0, 0.0),
        Point(0.5, 1.0, 0.0),
        Point(0.7, 1.2, 0.0),
        Point(1.5, 0.2, 0.0)
        ]
    table.triangles = [0, 1, 2, 0, 2, 3]

    on_table = []
    not_on_table = []
    for ii in range(300):
        p = np.random.random((2,))
        point = Point(p[0], p[1], 0.0)
        if point_on_table(point, table):
            on_table.append(p)
        else:
            not_on_table.append(p)

    vertices = [np.array((v.x, v.y)) for v in table.vertices]
    for tri_i in range(0, len(table.triangles), 3):
        i = table.triangles[tri_i]
        j = table.triangles[tri_i+1]
        k = table.triangles[tri_i+2]
        v_i = vertices[i]
        v_j = vertices[j]
        v_k = vertices[k]
        plt.plot([v_i[0], v_j[0], v_k[0], v_i[1]], [v_i[1], v_j[1], v_k[1], v_i[1]], 'g-')
    for p in on_table:
        plt.plot([p[0]], [p[1]], 'bo')
    for p in not_on_table:
        plt.plot([p[0]], [p[1]], 'ro')
    plt.show()
        

    
        
        
