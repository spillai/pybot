#!/usr/bin/python
import socket
import time
import cv2
import numpy as np

class TCPPub: 
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),90]        
    """
    TCP publisher class
    """
    def __init__(self): 
        self.server_uid_ = dict()
        self.get_uid = lambda ip, port: '{}:{}'.format(ip,port)

    def publish_image(self, im, ip='mrg-liljon.csail.mit.edu', port=12347): 
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

        im = cv2.resize(im, None, fx=0.5, fy=0.5)
        result, imgencode = cv2.imencode('.jpg', im, TCPPub.encode_param)
        data = np.array(imgencode)
        stringData = data.tostring()
        s.send( str(len(stringData)).ljust(16))
        s.send( stringData )

    def __del__(self): 
        for s in self.server_uid_.itervalues(): 
            s.close()

global g_tcp_pub
g_tcp_pub = TCPPub()

def publish_image(im, ip='mrg-liljon.csail.mit.edu', port=12347, flip_rb=True): 
    global g_tcp_pub
    g_tcp_pub.publish_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
                            if flip_rb else im)

