#!/bin/bash

# A single script that spawns a simple http.server, and
# a zeromq-based server that relays data to a websocket
# that the client browser can connect to.

echo "\033[92mGo to localhost:8000/viewer.html\033[00m";
trap "exit" INT TERM ERR
trap "kill 0" EXIT

python3 -m http.server &
python3 server.py &

wait

