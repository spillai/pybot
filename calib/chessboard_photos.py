#!/bin/env python
"""
Module for finding chessboards with a stereo rig.

python chessboard_photos.py --rows 8 --columns 6 --square-size 9.3 --calibration-folder calib_output 0 1 64 data

"""

import argparse
import calibrate_stereo
import os
import webcams

import cv2
import progressbar

import time
import numpy as np

from bot_vision.image_utils import to_color, im_resize, im_mosaic, im_pad
from bot_vision.imshow_utils import imshow_cv

class ChessboardFinder(webcams.BumblebeeStereoPair):
    """A ``StereoPair`` that can find chessboards."""

    def get_chessboard(self, columns, rows, show=False):
        """
        Take a picture with a chessboard visible in both captures.

        ``columns`` and ``rows`` should be the number of inside corners in the
        chessboard's columns and rows. ``show`` determines whether the frames
        are shown while the cameras search for a chessboard.
        """
        found_chessboard = [False, False]
        fvis = [None, None]
        while not all(found_chessboard):
            frames = self.get_frames()
            for i, frame in enumerate(frames):
                fvis[i] = to_color(frame.copy())
                (found_chessboard[i],
                corners) = cv2.findChessboardCorners(frame, (columns, rows),
                                                  flags=cv2.CALIB_CB_FAST_CHECK)
                cv2.drawChessboardCorners(fvis[i], (columns, rows), corners, found_chessboard[i])

            vis = np.hstack(fvis)
            label = np.tile(np.uint8([[[0,255,0]]]), (20, vis.shape[1], 1)) \
                   if all(found_chessboard) else np.tile(np.uint8([[[0,0,255]]]), (20, vis.shape[1], 1))
            imshow_cv('Checkerboard', im_resize(np.vstack([vis, label]), scale=0.75))
        return frames

PROGRAM_DESCRIPTION=(
"Take a number of pictures with a stereo camera in which a chessboard is "
"visible to both cameras. The program waits until a chessboard is detected in "
"both camera frames. The pictures are then saved to a file in the specified "
"output folder. After five seconds, the cameras are rescanned to find another "
"chessboard perspective. This continues until the specified number of pictures "
"has been taken."
)

def main():
    """
    Take a pictures with chessboard visible to both cameras in a stereo pair.
    """
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION,
                                parents=[calibrate_stereo.CHESSBOARD_ARGUMENTS])
    parser.add_argument("left", metavar="left", type=int,
                        help="Device numbers for the left camera.")
    parser.add_argument("right", metavar="right", type=int,
                        help="Device numbers for the right camera.")
    parser.add_argument("num_pictures", type=int, help="Number of valid "
                        "chessboard pictures that should be taken.")
    parser.add_argument("output_folder", help="Folder to save the images to.")
    parser.add_argument("--calibration-folder", help="Folder to save camera "
                        "calibration to.")
    parser.add_argument("--interval", type=float, default=1,
                        help="Interval (s) to take pictures in.")
    args = parser.parse_args()
    if (args.calibration_folder and not args.square_size):
        args.print_help()

    progress = progressbar.ProgressBar(maxval=args.num_pictures,
                                       widgets=[progressbar.Bar("=", "[", "]"),
                                                " ", progressbar.Percentage()])
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    # with ChessboardFinder((args.left, args.right)) as pair:

    # Mosaic calibration data
    mosaic = []
    imshow_cv('Calibration data', np.zeros(shape=(480,640)))
    # Interval captures
    pstamp = 0

    with ChessboardFinder() as pair:

        idx = 0
        while idx < args.num_pictures: 
            frames = pair.get_chessboard(args.columns, args.rows, show=False)

            # Only capture every interval seconds
            now = time.time()
            if now < pstamp + args.interval:
                continue
            pstamp = now

            for side, frame in zip(("left", "right"), frames):
                number_string = str(idx + 1).zfill(len(str(args.num_pictures)))
                filename = "{}_{}.ppm".format(side, number_string)
                output_path = os.path.join(args.output_folder, filename)
                cv2.imwrite(output_path, frame)
            idx += 1

            mosaic.append(im_pad(im_resize(np.hstack(list(frames)), scale=0.125), pad=3))
            progress.update(progress.maxval - (args.num_pictures - idx))
            imshow_cv('Calibration data', im_mosaic(*mosaic))
        progress.finish()
    if args.calibration_folder:
        args.input_files = calibrate_stereo.find_files(args.output_folder)
        args.output_folder = args.calibration_folder
        args.show_chessboards = True
        calibrate_stereo.calibrate_folder(args)

if __name__ == "__main__":
    main()
