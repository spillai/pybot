import cv2
import argparse, os
import numpy as np
import psutil
import joblib

def mkdir_p(path):
    try:
        os.makedirs(path)
    except : 
        pass 

def path_exists(path): 
    return os.path.exists(os.path.expanduser(path))

def create_path_if_not_exists(filename): 
    fn_path, fn_file = os.path.split(filename)    
    if not path_exists(fn_path): 
        mkdir_p(fn_path)
        return True
    return False

def joblib_dump(item, path): 
    if create_path_if_not_exists(path): 
        print 'Making directory for path %s' % path
    joblib.dump(item, path)
    
def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

class VideoWriter: 
    def __init__(self, filename): 
        create_path_if_not_exists(filename)
        self.filename = filename
        self.writer = None

    def __del__(self): 
        self.close()
        print 'Closing video writer and saving %s' % self.filename

    def write(self, im):
        if self.writer is None: 
            h, w = im.shape[:2]
            self.writer = cv2.VideoWriter(self.filename, cv2.cv.CV_FOURCC(*'mp42'), 
                                          25.0, (w, h), im.ndim == 3)
            print 'Creating writer: %s (%i,%i)' % (self.filename, w, h)
        self.writer.write(im)

    def close(self): 
        if self.writer is not None: 
            self.writer.release()

global g_fn_map
g_fn_map = {}
def write_video(fn, im): 
    global g_fn_map
    if fn not in g_fn_map: 
        g_fn_map[fn] = VideoWriter(fn)
    g_fn_map[fn].write(im)

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
