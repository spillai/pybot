# Copyright 2017 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

# Stripped from 
# https://github.com/googledatalab/pydatalab/blob/master/google/datalab/ml/_tensorboard.py

# try:
#   import IPython
# except ImportError:
#   raise Exception('This module can only be loaded in ipython.')

import os
import argparse
# import pandas as pd
import psutil
import subprocess
import time
from six.moves.http_client import HTTPConnection

# import google.datalab as datalab

def is_http_running_on(port):
    """ Check if an http server runs on a given port.
  Args:
    The port to check.
  Returns:
    True if it is used by an http server. False otherwise.
  """
    try:
      conn = HTTPConnection('127.0.0.1:' + str(port))
      conn.connect()
      conn.close()
      return True
    except Exception:
      return False

global g_tboard
g_tboard = None

def start_tensorboard(logdir, port=6006):
  global g_tboard
  g_tboard = TensorBoard.start(logdir, port=port)
  return g_tboard

def stop_tensorboard():
  global g_tboard
  if g_tboard:
      TensorBoard.stop(g_tboard)
      print('Stopping Tensorboard')

class TensorBoard(object):
  """Start, shutdown, and list TensorBoard instances.
  """

  @staticmethod
  def list():
    """List running TensorBoard instances.
    """
    running_list = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir')
    parser.add_argument('--port')
    for p in psutil.process_iter():
      if p.name() != 'tensorboard':
        continue
      cmd_args = p.cmdline()
      del cmd_args[0:2]  # remove 'python' and 'tensorboard'
      args = parser.parse_args(cmd_args)
      running_list.append({'pid': p.pid, 'logdir': args.logdir, 'port': args.port})
    return running_list
    # return pd.DataFrame(running_list)

  @staticmethod
  def start(logdir, port=6006):
    """Start a TensorBoard instance.

    Args:
      logdir: the logdir to run TensorBoard on.
    Raises:
      Exception if the instance cannot be started.
    """
    # if logdir.startswith('gs://'):
    #   # Check user does have access. TensorBoard will start successfully regardless
    #   # the user has read permissions or not so we check permissions here to
    #   # give user alerts if needed.
    #   datalab.storage._api.Api.verify_permitted_to_read(logdir)

    if is_http_running_on(port):
        print('Tensorboard already running')
        return None
    
    # port = datalab.utils.pick_unused_port()
    args = ['tensorboard', '--logdir=' + logdir, '--port=' + str(port)]
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(args, stdout=FNULL, stderr=FNULL)
    retry = 5
    while (retry > 0):
      if is_http_running_on(port):
        # url = '/_proxy/%d/' % port
        # html = '<p>TensorBoard was started successfully with pid %d. ' % p.pid
        # html += 'Click <a href="%s" target="_blank">here</a> to access it.</p>' % url
        # IPython.display.display_html(html, raw=True)
        return p.pid
      time.sleep(1)
      retry -= 1

    raise Exception('Cannot start TensorBoard.')

  @staticmethod
  def stop(pid):
    """Shut down a specific process.

    Args:
      pid: the pid of the process to shutdown.
    """
    if psutil.pid_exists(pid):
      try:
        p = psutil.Process(pid)
        p.kill()
      except Exception:
        pass
