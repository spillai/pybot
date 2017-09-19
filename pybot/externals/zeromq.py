import zmq
import time

global g_context
global g_socket

g_context = None
g_socket = None
def connect(server='127.0.0.1', port=4999):
    global g_context
    global g_socket
    try: 
        g_context = zmq.Context()
        g_socket = g_context.socket(zmq.PUB)
        g_socket.setsockopt(zmq.LINGER, 0)
        g_socket.bind("tcp://{}:{}".format(server, port))
        print('Connected successfully to {}:{}'.format(server, port))

        # Sleep a little bit so that packets are not dropped
        time.sleep(0.2)
    except Exception, e:
        g_context, g_socket = None, None
        print('Exception {}'.format(e))
    
    
def pack(channel, data):
    return channel + b' ' + data

def unpack(msg):
    channel, data = msg.split(b' ', 1)
    return channel, data

def publish(channel, data):
    global g_context
    global g_socket
    if g_context is None:
        connect()
    g_socket.send(pack(channel, data))
