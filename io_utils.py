import cv2
import sys
import argparse
import ConfigParser
import os
import numpy as np
import psutil
import joblib
from cStringIO import StringIO
from bot_vision.image_utils import im_resize

def format_time(t):
    if t > 60:
        return "%5.2fmin" % (t / 60.)
    else:
        return " %6.2fs" % (t)

def mkdir_p(path):
    try:
        os.makedirs(os.path.expanduser(path))
    except : 
        print 'Failed to create path %s' % path 

def path_exists(path): 
    return os.path.exists(os.path.expanduser(path))

def create_directory_if_not_exists(dir_path): 
    """ Create directory path if it doesn't exist """
    
    if not path_exists(dir_path): 
        mkdir_p(dir_path)
        return True
    return False

def create_path_if_not_exists(filename): 
    """ Create directory path if it doesn't exist """
    
    fn_path, fn_file = os.path.split(filename)    
    if not path_exists(fn_path): 
        mkdir_p(fn_path)
        return True
    return False

def number_of_files(directory, ext=''): 
    return len(filter(lambda x: ext in x, [item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))]))

def read_config(conf_path, section): 
    """
    Recipe mostly taken from
    http://blog.vwelch.com/2011/04/combining-configparser-and-argparse.html 
    """
    # Try reading the conf file
    try: 
        config = ConfigParser.SafeConfigParser()
        config.read([conf_path])
        defaults = dict(config.items(section))
    except Exception as e: 
        raise RuntimeError('Failed reading %s: %s' % (conf_path, e))

    return defaults


def config_and_args_parser(conf_path, section, description=''): 
    """
    Recipe mostly taken from
    http://blog.vwelch.com/2011/04/combining-configparser-and-argparse.html 
    """
    
    # Parse directory
    parser = argparse.ArgumentParser(
        description=description)
    parser.add_argument('-c', '--conf_file', required=False, 
                        default=conf_path, help='Specify config file', metavar='FILE')
    args, remaining_argv = parser.parse_known_args()

    # Try reading the conf file
    try: 
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults = dict(config.items(section))
    except Exception as e: 
        raise RuntimeError('Failed reading %s: %s' % (args.conf_file, e))

    parser.set_defaults(**defaults)
    # parser.add_argument("--option1", help="some option")
    args = parser.parse_args(remaining_argv)        
    return args


def joblib_dump(item, path): 
    create_path_if_not_exists(path)
    joblib.dump(item, path)
    
def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

class Tee(object):
    """ General purpose tee-ing object that allows to split  stdout """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self): 
        pass

def tee_stdout(filename): 
    """ Redirect output to log, and stderr """

    create_path_if_not_exists(filename)
    log_file = open(filename, 'w', 0)
    orig_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

class VideoWriter: 
    def __init__(self, filename, as_images=False): 
        if as_images: 
            directory = os.path.join(''.join([filename, '-imgs']))
            create_path_if_not_exists(os.path.join(directory, 'movie.avi'))
        else: 
            directory = None
            create_path_if_not_exists(filename)

        self.as_images = as_images
        self.filename = filename
        self.directory = directory
        self.writer = None
        self.idx = 0

    def __del__(self): 
        self.close()
        print 'Closing video writer and saving %s' % self.filename

    def write(self, im): 
        if self.as_images: 
            return self._write_images(im)
        else: 
            return self._write_video(im)

    def _write_images(self, im): 
        cv2.imwrite(os.path.join(self.directory, 'imgs-%06i.png' % self.idx), im)
        self.idx += 1

    def _write_video(self, im):
        if self.writer is None: 
            h, w = im.shape[:2]
            self.writer = cv2.VideoWriter(self.filename, cv2.cv.CV_FOURCC(*'mp42'), 
                                          30.0, (w, h), im.ndim == 3)
            print 'Creating writer: %s (%i,%i)' % (self.filename, w, h)
        self.writer.write(im)

    def close(self): 
        if self.as_images: 
            # ffmpeg?
            pass
        if self.writer is not None: 
            self.writer.release()

global g_fn_map
g_fn_map = {}
def write_video(fn, im, scale=1.0, as_images=False): 
    global g_fn_map
    if fn not in g_fn_map: 
        g_fn_map[fn] = VideoWriter(fn, as_images=as_images)
    im_scaled = im_resize(im, scale=scale) if scale != 1.0 else im
    g_fn_map[fn].write(im_scaled)

# def write_images(template, im): 

import subprocess
class VideoSink(object) :
    def __init__( self, size, filename="output", rate=10, byteorder="bgra" ) :
            self.size = size
            cmdstring  = ('mencoder',
                   '/dev/stdin',
                    '-demuxer', 'rawvideo',
                          '-rawvideo', 'w=%i:h=%i'%size[::-1]+":fps=%i:format=%s"%(rate,byteorder),
                    '-o', filename+'.avi',
                    '-ovc', 'lavc',
                    )
            self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

    def run(self, image) :
        assert image.shape[:2] == self.size
        self.p.stdin.write(image.tostring())
    def close(self) :
            self.p.stdin.close()            

class VideoCapture(object): 
    def __init__(self, filename=-1, fps=30, size=(640,480), process_cb=None): 
        self.cap = cv2.VideoCapture(filename)

        if isinstance(filename, int): 
            if size is not None: 
                self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, size[0])
                self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, size[1])
            if fps is not None: 
                self.cap.set(cv2.cv.CV_CAP_PROP_FPS, fps)

            # self.cap.set(cv2.cv.CV_CAP_PROP_MODE, 70) # MODE_640x480_MONO16

        self.process_cb = process_cb

    def get(self): 
        ret, im = self.cap.read()
        if not ret: 
            raise RuntimeError('Failed to read image from capture device')
        return im

    def iteritems(self): 
        while True: 
            try: 
                ret, im = self.cap.read()
                if not ret: break
                yield im
            except KeyboardInterrupt: 
                break
        

    def run(self): 
        assert(self.process_cb is not None)
        while True: 
            try: 
                ret, im = self.cap.read()
                if not ret: break
                self.process_cb(im)
            except KeyboardInterrupt: 
                break

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        usage='python video_writer.py -t <template> -o <output_filename>'
    )
    parser.add_argument('--template', '-t', 
                        help="Template filename e.g. image_%06i")
    parser.add_argument('--output', '-o', 
                        help="Ouptut filename")
    parser.add_argument('--range', '-r', default='0-10000',
                        help="Index range e.g. 0-1000")
    (options, args) = parser.parse_known_args()
    
    # Required options =================================================
    if not options.template or not options.output: 
        parser.error('Output Filename/Template not given')

    start, end = map(int, options.range.split('-'))
    VideoWriter(filename=options.output, template=options.template, start_idx=start, max_files=end-start).run()
