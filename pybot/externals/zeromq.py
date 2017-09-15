import zmq

global g_context
global g_socket

g_context = zmq.Context()
g_socket = g_context.socket(zmq.PUB)
g_socket.bind("tcp://127.0.0.1:4999")

def publish(channel, data):
    global g_socket
    g_socket.send(channel + b' ' + data)
    
def receive(msg):
    ch, data = msg.split(b' ', 1)
    return ch, data
            
