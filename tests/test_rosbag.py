#!/usr/bin/env python

import os
import time
import argparse
import cv2
import numpy as np

from collections import deque
from pybot.utils.itertools_recipes import take

from pybot.vision.camera_utils import DepthCamera
from pybot.externals.ros.bag_utils import ROSBagReader, ROSBagController
from pybot.externals.ros.bag_utils import ImageDecoder, NavMsgDecoder, TfDecoderAndPublisher
from pybot.vision.image_utils import to_color
from pybot.vision.imshow_utils import imshow_cv, print_status
from pybot.externals.lcm.draw_utils import publish_pose_list, publish_cloud

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(
        description='rosbag player')
    parser.add_argument(
        '-f', '--filename', type=str, required=True, 
        help="Filename: rosbag (.bag)")
    parser.add_argument(
        '-c', '--camera-channel', type=str, required=False, 
        default='/camera/rgb/image_raw', 
        help='/camera/rgb/image_raw')
    parser.add_argument(
        '-d', '--depth-channel', type=str, required=False, 
        default='/camera/depth/image_raw', 
        help='/camera/rgb/image_raw')
    parser.add_argument(
        '-o', '--odom-channel', type=str, required=False, 
        default='/odom', 
        help='/odom')
    args = parser.parse_args()

    # Setup dataset/log
    dataset = ROSBagReader(filename=os.path.expanduser(args.filename), 
                           decoder=[
                               ImageDecoder(channel=args.camera_channel, scale=1, 
                                            compressed='compressed' in args.camera_channel), 
                               ImageDecoder(channel=args.depth_channel, scale=1, encoding='passthrough',
                                            compressed='compressed' in args.camera_channel), 
                               NavMsgDecoder(channel=args.odom_channel, every_k_frames=10),
                           ],                                    
                           every_k_frames=1, start_idx=0, index=False)

    # Iterate through rosbag reader
    odom = deque(maxlen=100)
    for idx, (t, ch, data) in enumerate(take(dataset.iterframes(), 10)): 
        if ch == args.odom_channel:
            odom.append(data)
            publish_pose_list('robot_poses', odom, frame_id='origin', reset=(idx == 0))
            continue
        imshow_cv('left', data)
    print('Read {} frames'.format(idx+1))

    # Camera calibration
    cam, = dataset.calib(['/camera/depth/camera_info'])

    global dcam
    dcam = DepthCamera(K=cam.K)

    # Define callbacks
    global odom_str, rgb
    rgb = None
    odom_str = ''
    def on_image_cb(t, im):
        global rgb
        vis = to_color(im)
        rgb = to_color(im)[::4,::4]
        vis = print_status(vis, text='Pose: {}'.format(odom_str))
        imshow_cv('im', vis)

    def on_depth_cb(t, im):
        global rgb
        if rgb is None: 
            return
        X = dcam.reconstruct(im)[::4,::4,:]
        publish_cloud('depth', X, c=rgb, frame_id='camera', flip_rb=True)
        imshow_cv('d', im / 10.0)
        
    def on_odom_cb(t, data): 
        global odom_str
        odom_str = '{}, {}'.format(t, data)

    # Create ROSBagController
    controller = ROSBagController(dataset)
    controller.subscribe(args.camera_channel, on_image_cb)
    controller.subscribe(args.odom_channel, on_odom_cb)
    controller.subscribe(args.depth_channel, on_depth_cb)
    controller.run()
