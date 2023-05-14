"""
Command-Line script used to compute ARI (Adapted Rand-Index) error for input segmentations

It takes as parameters:

1. paths to segmentations to evaluate
2. path to ground truth segmentation
3. path to output file where results will be stored

Then:

1. opens the input segmentations and the ground truth segmentation
2. calls the subroutine in the segmentation.py module
3. prints output and stores to specified output file
"""

import argparse
import h5py
import imageio
import numpy as np
import lioness.segmentation.segmentation as Segmentation


parser = argparse.ArgumentParser()
parser.add_argument("--seg", help="Paths to segmentations", nargs="+", required=True)
parser.add_argument("--gt", help="Paths to ground truth volume", required=True)
parser.add_argument("--out", help="Paths to result files", nargs="+", required=True)
args = parser.parse_args()

if __name__ == "__main__":
    segs = [Segmentation.from_file(fpath) for fpath in args.seg]
    gt = Segmentation.from_file(args.gt)
    aris = [seg.adapted_rand(gt.data) for seg in segs]
    for i in range(len(aris)):
        fout = open(args.out[i], "w")
        fout.write("%f" % aris[i])
        fout.close()
