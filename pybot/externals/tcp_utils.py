# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import socket
import time
import cv2
import numpy as np
from pybot.externals.print_utils import print_green

def recvall(conn, count):
    buf = b''
    while count:
        newbuf = conn.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def read_image(conn): 
    try: 
        length = int(recvall(conn, 16))
    except:
        import sys
        print "Unexpected error:", sys.exc_info()[0]
        return False, None

    stringData = recvall(conn, length)
    data = np.fromstring(stringData, dtype='uint8')
    decimg = cv2.imdecode(data, 1)
    print 'Image received ', decimg.shape
    return True, decimg


def send_image(s, im, scale=1.0, encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]):
    # Will not upsample image for bandwidth reasons
    if scale < 1: 
        im = cv2.resize(im, None, fx=scale, fy=scale)
    result, imgencode = cv2.imencode('.jpg', im, encode_param)
    data = np.array(imgencode)
    stringData = data.tostring()
    s.send( str(len(stringData)).ljust(16))
    s.send( stringData )


class TCPServer(object):
    def __init__(self, ip='', port=12347):
        self.ip_, self.port_ = ip, port
        self.init(self.ip_, self.port_)
        
    def init(self, ip, port): 
        self.s_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s_.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print_green('Hostname: {:} READY'.format(socket.gethostname()))
        
        self.s_.bind((self.ip_, self.port_))
        self.s_.listen(1)
        self.conn_, self.addr_ = self.s_.accept()

    def _read(self):
        return read_image(self.conn_)

    def read(self):
        rval, im = self._read()
        if not rval:
            self.release()
            print_green('Waiting for new connection')
            self.init(self.ip, self.port)
            rval, im = self._read()
        return im

    def on_image(self, im):
        raise NotImplementedError()

    def run(self):
        rval = True

        while rval:
            start = time.time()
            rval, im = cap._read()
            end = time.time()

            if not rval:
                self.release()
                print_green('Waiting for new connection')
                self.init(self.ip_, self.port_)
                rval = True
                continue

            self.on_image(self, im)
    
    def release(self):
        self.s_.close()

    @property
    def ip(self):
        return self.ip_

    @property
    def port(self):
        return self.port_
        
class TCPControl(object):
    def __init__(self, ip='', port=12347):
        self.server_ = TCPServer(ip=ip, port=port)
        
    def __enter__(self):
        return self.server_
    
    def __exit__(self, type, value, traceback):
        self.server_.release()

class TCPPub: 

    """
    TCP publisher class
    """
    def __init__(self): 
        self.server_uid_ = dict()
        self.get_uid = lambda ip, port: '{}:{}'.format(ip,port)

    def publish_image(self, im, ip='', port=12347, scale=1.0): 
        s = None
        suid = self.get_uid(ip,port)
        if suid not in self.server_uid_: 
            try: 
                s = socket.socket()
                s.connect((ip, port))
                self.server_uid_[suid] = s
            except: 
                print('Unavailable TCP server {:}:{:}'.format(ip,port))            
        else: 
            s = self.server_uid_[suid]

        send_image(s, im, scale=scale)

    def __del__(self): 
        for s in self.server_uid_.itervalues(): 
            s.close()

global g_tcp_pub
g_tcp_pub = TCPPub()

def publish_image(im, ip='', port=12347, flip_rb=True, scale=1.0): 
    global g_tcp_pub
    g_tcp_pub.publish_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
                            if flip_rb else np.copy(im), scale=scale)
    time.sleep(0.1)
