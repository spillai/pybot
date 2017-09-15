import lcm

import multiprocessing as mp
from websocket_server import WebsocketServer

# Called for every client connecting (after handshake)
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    server.send_message_to_all("Hey all, a new client has joined us")

# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])

# Called when a client sends a message
def message_received(client, server, message):
    if len(message) > 200:
        message = message[:200]+'..'
    print("Client(%d) said: %s" % (client['id'], message))
        
class LCMThreadHandler():
    def __init__(self, server):
        self.server_ = server

        self.lc_th_ = mp.Process(target=self.run)
        self.lc_th_.start()
        
    def on_lcm_event(self, ch, data):
        print('lcm_event: {}, {}'.format(ch, len(data)))
        self.server_.send_message_to_all(data)

    def run(self):

        # Setup
        self.lc_ = lcm.LCM()
        self.sub_ = self.lc_.subscribe('.*_COLLECTION.*', self.on_lcm_event)

        # Handler
        try:
            while True:
                self.lc_.handle()
        except KeyboardInterrupt:
            pass

        # Join 
        self.lc_th.join()
        
        
PORT=9001
server = WebsocketServer(PORT)
print('Starting server on port {}'.format(PORT))
lcm_th = LCMThreadHandler(server)

server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
server.set_fn_message_received(message_received)
server.run_forever()
