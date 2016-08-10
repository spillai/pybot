import cv2
import numpy as np
import shapely.geometry as sg

from bot_vision.imshow_utils import imshow_cv
from bot_vision.color_utils import color_from_string, get_color_by_label

def reduce_funcs(*funcs): 
    from functools import wraps
    @wraps
    def wrapper(*args): 
        return reduce(funcs, args)
    return wrapper

class MethodDecorator(object):
    """
    Decorates class methods with wrapped methods. 
    Apply func=to_polygon to each of the methods [intersection, union] etc. 
    such that the returned value is chained resulting in to_polygon(intersection(x))
    """
    def __init__(self, func, methods):
        self.func = func
        self.methods = methods
    def __call__(self, cls):
        class Wrapped(cls):
            for attr in self.methods:
                if hasattr(cls, attr):
                    setattr(cls, attr, reduce_funcs(getattr(cls, self.func), getattr(cls, attr)))
        return Wrapped

@MethodDecorator(func='to_polygon', methods=('intersection', 'union'))
class Polygon(sg.Polygon): 
    """
    Accessible attributes: area etc.
    """
    def __init__(self, pts=None, pg=None): 
        assert(pts is not None or pg is not None)
        if pg is not None: 
            sg.Polygon.__init__(self, pg)
        else: 
            sg.Polygon.__init__(self, pts.tolist())

    @classmethod
    def to_polygon(cls, pg): 
        return cls(pg=pg)

    @property 
    def pts(self): 
        return np.array(self.exterior.coords)

    @property
    def center(self):
        return np.array([self.centroid.x, self.centroid.y])

    @property
    def width(self): 
        return self.bounds[2]-self.bounds[0]

    @property
    def height(self): 
        return self.bounds[2]-self.bounds[0]

    @property
    def size(self):
        return self.width, self.height

    def percentoverlap(self, other): 
        return self.intersection(other).area / self.union(other).area

    def contains(self, pt):
        return sg.Point(pt[0], pt[1])

    def resize(self, xratio, yratio = None):
        if yratio is None:
            yratio = xratio
        c, pts = self.center, self.pts
        dv = pts - c
        mag = np.linalg.norm(dv, axis=1)
        return Polygon(np.hstack([c[0] + dv[:,0] * mag * xratio, 
                                  c[0] + dv[:,0] * mag * xratio]))

class Box(Polygon): 
    def __init__(self, bounds=None, box=None): 
        assert(bounds is not None or box is not None)
        if box is not None: 
            sg.box.__init__(self, box)
        else: 
            if type(bounds) == np.ndarray: 
                bounds = bounds.tolist()
            sg.box.__init__(self, bounds)

    @classmethod
    def from_pts(cls, pts): 
        xmin, xmax = np.min(pts[:,0]), np.max(pts[:,0])
        ymin, ymax = np.min(pts[:,1]), np.max(pts[:,1])
        return cls(bounds=[xmin, ymin, xmax, ymax])

class Annotator(object): 
    def __init__(self, im, name, annotation={}): 
        win_name = "Annotator - %s" % name
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self._on_mouse)

        # self._print_hotkeys()
        self.im = im.copy()
        self.name = win_name
        self.reset()

        self.pts_map = dict(annotation)
        print annotation
        self.show()

    def _print_hotkeys(self): 
        print '''
        Annotator: 
        \tr - reset
        \tn - next id
        \tp - previous id
        \ts - save
        \tEsc - quit
        '''

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pt = np.array([[x, y]], dtype=np.int)

        id_key = 'id_%i' % self.pt_id
        if id_key in self.pts_map: 
            self.pts_map[id_key] = np.vstack([self.pts_map[id_key], pt])
        else: 
            self.pts_map[id_key] = pt
        self.show()
        
    def show(self): 
        im = self.im.copy()

        colors = get_color_by_label(np.arange(10)) * 255
        for pid, pts in enumerate(self.pts_map.itervalues()): 
            col = tuple(map(int, colors[pid % len(colors)].ravel())) if im.ndim == 3 else 255
            for pt in pts: 
                cv2.circle(im, tuple(map(int, pt)), 2, col, -1, lineType=cv2.CV_AA)
            cv2.polylines(im, [pts.reshape(-1,1,2)], 1, col, thickness=2)       
        imshow_cv(self.name, im)

    def reset(self): 
        self.pts_map = dict()
        self.pt_id = 0
        self.show()

    def run(self): 
        annotate = True
        while True:
            ch = 0xFF & cv2.waitKey(1)
            if ch == ord('r'):
                self.reset()
            elif ch == ord('n'): 
                self.pt_id += 1
            elif ch == ord('p'): 
                self.pt_id -= 1
            elif ch == ord('s'): 
                print 'Saving to %s' % self.name
                self.pts_map.save()
                pass
                # savemat
            elif ch == ord('f'): 
                break
            elif ch == ord('q') or ch == 27: 
                annotate = False
                break

        cv2.destroyWindow(self.name)
        return annotate, self.pts_map

if __name__ == "__main__": 
    pts = np.array([[0,0], [0,1], [1,1], [1.5, 0.5], [1,0]])
    pts2 = np.array([[0.5,0.5], [0.5,1.5], [1.5,1.5], [1.5,0.5]])
    a = Polygon(pts)
    print a.width
    b = Polygon(pts2)

    print a.pts
    print a.intersection(b).pts

    
