import multiprocessing as mp
from threading import Lock, RLock

from websocket_server import WebsocketServer

from pybot.externals import marshalling_backend
from pybot.externals import unpack, pack

class _ThreadHandler(object):
    def __init__(self):
        self.lock_ = Lock()
        self.ev_th_ = None
        
    def setup(self, server):
        self.ev_th_ = mp.Process(target=self.run, args=(server,))
        self.ev_th_.start()
        with self.lock_:
            self.server_ = server
            
    def stop(self):
        try: 
            self.ev_th_.join()
        except Exception as e:
            print('Exiting')
            
    def on_event(self, server, msg):
        try:
            ch, data = unpack(msg)
            print('on_event: ch={}, len={}'.format(ch, len(data)))
            server.send_message_to_all(msg)
        except Exception as e:
            print('Failed to send, client unavailable {}'.format(e))            
        
    # Called for every client connecting (after handshake)
    def new_client(self, client, server):
        self.setup(server)
        with self.lock_:
            print("New client connected and was given id %d" % client['id'])
            # self.server_.send_message_to_all("Hey all, a new client has joined us")

    # Called for every client disconnecting
    def client_left(self, client, server):
        self.setup(server)
        with self.lock_:
            print("Client(%d) disconnected" % client['id'])

    # Called when a client sends a message
    def message_received(self, client, server, message):
        if len(message) > 200:
            message = message[:200]+'..'
        print("Client(%d) said: %s" % (client['id'], message))
        
    def run(self, server):

        # Setup
        if marshalling_backend() == 'lcm':
            import lcm
            self.m_ = lcm.LCM()
            self.sub_ = self.m_.subscribe('.*_COLLECTION.*', self.on_event)
                
            def handle():
                # Handler
                try:
                    while True:
                        self.lc_.handle()
                except KeyboardInterrupt:
                    pass

            def cleanup():
                pass
            
            
        elif marshalling_backend() == 'zmq':
            import zmq

            zmq_server = '127.0.0.1'
            zmq_port = 4999
            self.m_ = zmq.Context()
            self.sub_ = self.m_.socket(zmq.SUB)
            self.sub_.connect('tcp://{}:{}'
                              .format(zmq_server, zmq_port))
            self.sub_.setsockopt(zmq.SUBSCRIBE, b'')
            print('Starting zmq listener on port {}:{}'
                  .format(zmq_server, zmq_port))
            
            def handle():
                # Handler
                try:
                    while True:
                        msg = self.sub_.recv()
                        self.on_event(server, msg)
                except KeyboardInterrupt:
                    pass
            
            def cleanup():
                self.sub_.close()
                self.m_.term()


        # Handle
        handle()
        

        
PORT=9001
th = _ThreadHandler()

print('Starting server on port {}'.format(PORT))
server = WebsocketServer(PORT)
server.set_fn_new_client(th.new_client)
server.set_fn_client_left(th.client_left)
server.set_fn_message_received(th.message_received)

server.run_forever()
th.stop()
