#!/bin/bash

echo "Go to localhost:8000/viewer.html";
trap "exit" INT TERM ERR
trap "kill 0" EXIT

python3 -m http.server &
python3 server.py &

wait

