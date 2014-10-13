#! /usr/bin/env python

# Imdescrip -- a collection of tools to extract descriptors from images.
# Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import glob, sys, os
import cPickle as pk
import argparse
from imdescrip.extractor import extract_smp
from imdescrip.descriptors.ScSPM import ScSPM

parser = argparse.ArgumentParser(description="Create a ScSPM dictionary.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("imagedir", help="Directory of training images.")
parser.add_argument("extension", help="Image file extension (eg. 'png').")
parser.add_argument("--dicname", help="Name and path of dictionary file to " 
                    "save.", default="ScSPM.p")
parser.add_argument("--nbases", help="Number of dictionary bases.", type=int, 
                    default=512)
parser.add_argument("--dcompress", help="Number of dimensions to compress "
                    "features to.", type=int, default=3000)
parser.add_argument("--npatches", help="Number of image patches to use to learn"
                    " dictionary.", type=int, default=200000)
args = parser.parse_args()

# Make a list of images
filelist = glob.glob(os.path.join(args.imagedir, '*.' + args.extension))

nimages = len(filelist)
print "Found {0} images.".format(nimages)
if nimages == 0:
    print "Quiting..."
    sys.exit(1)

# Train a dictionary
desc = ScSPM(dsize=args.nbases, compress_dim=args.dcompress)
desc.learn_dictionary(filelist, npatches=args.npatches, niter=5000)

# Save the dictionary
with open(args.dicname, 'wb') as f:
    pk.dump(desc, f, protocol=2)

print "Done! Dictionary object saved to {0}.".format(args.dicname)

