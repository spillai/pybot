import pprint
import argparse

from calibrate_stereo import get_stereo_calibration_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print camera calibration")
    parser.add_argument("input_folder", help="Input folder containing calibration")
    args = parser.parse_args()
    
    print pprint.pprint(get_stereo_calibration_params(input_folder=args.input_folder))

