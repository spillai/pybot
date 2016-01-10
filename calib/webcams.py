#!/bin/env python
"""A base class for working with stereo cameras."""

import argparse
import os
import time
import numpy as np

import cv2
from bot_vision.image_utils import to_gray
from bot_vision.imshow_utils import imshow_cv

# from pybot_drivers import DC1394Device, ZEDDevice

class StereoPair(object):
    """
    A stereo pair of cameras.

    Should be initialized with a context manager to ensure that the cameras are
    freed properly after use.
    """

    def __init__(self, name, devices):
        """
        Initialize cameras.

        ``devices`` is an iterable containing the device numbers.
        """
        #: Video captures associated with the ``StereoPair``
        self.captures = [cv2.VideoCapture(device) for device in devices]
        
        # for cap in self.captures: 
        #     cap.set(cv2.cv.CV_CAP_PROP_FPS, 10)
        # print('Setting capture rate to 10 Hz')

        #: Window names for showing captured frame from each camera
        self.windows = ["{} camera".format(side) for side in ("Left", "Right")]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for capture in self.captures:
            capture.release()
        for window in self.windows:
            cv2.destroyWindow(window)

    def get_frames(self):
        """Get current frames from cameras."""
        return [cv2.transpose(capture.read()[1]) if idx == 0 else 
                cv2.flip(cv2.transpose(capture.read()[1]), -1) for idx, capture in enumerate(self.captures)]

    # def get_frames(self):
    #     """Get current frames from cameras."""
    #     return [capture.read()[1] for idx, capture in enumerate(self.captures)]

    def show_frames(self, wait=0):
        """
        Show current frames from cameras.

        ``wait`` is the wait interval before the window closes.
        """
        for window, frame in zip(self.windows, self.get_frames()):
            cv2.imshow(window, frame)
        cv2.waitKey(wait)

    def show_videos(self):
        """Show video from cameras."""
        while True:
            self.show_frames(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class CustomStereoPair(object):
    """
    A stereo pair of cameras.

    Should be initialized with a context manager to ensure that the cameras are
    freed properly after use.
    """

    def __init__(self, name='bb', devices=None):
        """
        Initialize cameras.

        ``devices`` is an iterable containing the device numbers.
        """
        #: Video captures associated with the ``StereoPair``
        if name == 'bb': 
            self.capture = DC1394Device()
            self.capture.init()
        elif name == 'zed-sdk': 
            self.capture = ZEDDevice('720')
            self.capture.init()
        elif name == 'zed': 
            # try: 
            self.capture = cv2.VideoCapture(int(devices))

            # except: 
            #     raise RuntimeError('Failed to open ZED camera via UVC')
        else: 
            raise RuntimeError('Unknown stereo camera name: %s' % name)

        #: Window names for showing captured frame from each camera
        self.windows = ["{} camera".format(side) for side in ("Left", "Right")]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def get_frames(self):
        """Get current frames from cameras."""
        l,r = self.capture.getImages()
        return to_gray(l), to_gray(r)

    def show_frames(self, wait=0):
        """
        Show current frames from cameras.

        ``wait`` is the wait interval before the window closes.
        """
        imshow_cv('stereo', np.hstack([frame for window, frame in zip(self.windows, self.get_frames())]))

    def show_videos(self):
        """Show video from cameras."""
        while True:
            self.show_frames(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class BaseStereoPair(object): 
    def __init__(self): 
        """
        Base stereo pair
        Setup windows and basic stereo pair utilities
        """

        #: Window names for showing captured frame from each camera
        self.windows = ["{} camera".format(side) for side in ("Left", "Right")]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def show_frames(self, wait=0):
        """
        Show current frames from cameras.

        ``wait`` is the wait interval before the window closes.
        """
        imshow_cv('stereo', np.hstack([frame for window, frame in zip(self.windows, self.get_frames())]))

    def show_videos(self):
        """Show video from cameras."""
        while True:
            self.show_frames(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def get_frames(self):
        """Get current frames from cameras."""
        raise NotImplementedError    

class ZEDStereoPair(BaseStereoPair): 
    def __init__(self, device=None): 
        """
        Initialize ZED stereo pair from UVC or via SDK.

        ``devices`` is an iterable containing the device numbers.
        """
        BaseStereoPair.__init__(self)

        #: Video captures associated with the ``StereoPair``
        if device is None:
            self.capture = ZEDDevice('720')
            self.capture.init()
        else: 
            try: 
                self.capture = cv2.VideoCapture(int(device))
            except: 
                raise RuntimeError('Failed to open ZED camera via UVC')

    def get_frames(self):
        """Get current frames from cameras."""
        ret, im = self.capture.read()
        l, r = np.split(im, 2, axis=1)
        return l, r

def main():
    """
    Show the video from two webcams successively.

    For best results, connect the webcams while starting the computer.
    I have noticed that in some cases, if the webcam is not already connected
    when the computer starts, the USB device runs out of memory. Switching the
    camera to another USB port has also caused this problem in my experience.
    """
    parser = argparse.ArgumentParser(description="Show video from two "
                                     "webcams.\n\nPress 'q' to exit.")
    parser.add_argument("devices", type=int, nargs=2, help="Device numbers "
                        "for the cameras that should be accessed in order "
                        " (left, right).")
    parser.add_argument("--output_folder",
                        help="Folder to write output images to.")
    parser.add_argument("--interval", type=float, default=1,
                        help="Interval (s) to take pictures in.")
    args = parser.parse_args()

    with ZEDStereoPair(device=args.devices[0]) as pair: 
    # with CustomStereoPair(name='zed', devices=args.devices[0]) as pair:
    # with StereoPair('webcam', args.devices) as pair:
        if not args.output_folder:
            pair.show_videos()
        else:
            i = 1
            while True:
                start = time.time()
                while time.time() < start + args.interval:
                    pair.show_frames(1)
                images = pair.get_frames()
                for side, image in zip(("left", "right"), images):
                    filename = "{}_{}.ppm".format(side, i)
                    output_path = os.path.join(args.output_folder, filename)
                    cv2.imwrite(output_path, image)
                    print output_path
                i += 1

if __name__ == "__main__":
    main()
