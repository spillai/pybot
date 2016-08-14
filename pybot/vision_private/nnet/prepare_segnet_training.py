import fnmatch
import os
import argparse
from itertools import izip

import numpy as np
from sklearn.cross_validation import train_test_split

def write_data(filename, image_files, target_files): 
    with open(filename, 'w') as f: 
        for (image_f, target_f) in izip(image_files, target_files): 
            f.write('{} {}\n'.format(image_f, target_f))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create training file (train.txt)')
    parser.add_argument(
        '-d', '--directory', type=str, 
        default='', required=True, 
        help="Directory")
    parser.add_argument(
        '-f', '--filename', type=str, 
        default='', required=True, 
        help="Output filename")
    args = parser.parse_args()

    files = [
        os.path.join(args.directory, f)
        for f in os.listdir(os.path.expanduser(args.directory)) 
        if fnmatch.fnmatch(f, '*_raw.png')]

    image_files, target_files = [], []
    for f in files: 
        target_f = f.replace('raw.png', 'synth.png')
        if not os.path.exists(f) or not os.path.exists(target_f): 
            continue
        image_files.append(os.path.basename(f))
        target_files.append(os.path.basename(target_f))
        if len(image_files) % 100 == 0: 
            print('Found {} files for training'.format(len(image_files)))
    print('Found {} files for training'.format(len(image_files)))

    train_image_files, test_image_files, \
        train_target_files, test_target_files = train_test_split(image_files, target_files, train_size=0.7, random_state=0)
    write_data(args.filename +  'train.txt', train_image_files, train_target_files)
    write_data(args.filename + 'test.txt', test_image_files, test_target_files)
    print('Written to file {} with {} entries'.format(args.filename, len(image_files)))
