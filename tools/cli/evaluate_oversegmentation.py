"""
Command-Line script used to evaluate oversegmentation of segmentations produced by the automatic method
in the form of different metrics:

- ratio of segments in the automatic segmentation vs the ground-truth segmentation
- statistics on segment split count comparing segments in ground-truth vs overlapping segments in automatic segmentation
- number of small segments in automatic segmentation


It takes as parameters:

1. path to ground truth segmentation
2. paths to segmentations produced by the automatic method
3. threshold determining segment overlaps as segment splits
4. absolute and relative thresholds used to define "small segments"
5. path to output file where data will be stored

Then:

1. opens the automatic and the ground truth segmentations
2. calls the subroutine in the oversegmentation.py module
3. prints output and stores to specified output file
"""

import argparse
import numpy as np
import pandas as pd
import pathlib
from scipy import ndimage as ndi
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import seaborn as sns
from lioness.utils import utils as Utils
from lioness.segmentation import segmentation as Segmentation
from lioness.segmentation import oversegmentation as Oversegmentation


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gtseg", help="Path to file containing ground-truth segmentation"
    )
    parser.add_argument(
        "ossegs", nargs="+", help="Path to files containing other segmentations"
    )

    parser.add_argument(
        "-sst",
        "--segment_split_threshold",
        type=float,
        help="Threshold to consider overlap as segment split",
        default=0.8,
    )

    parser.add_argument(
        "-sca",
        "--size_count_abs_threshold",
        type=int,
        help="Size threshold (absolute) to count small segments",
        default=20,
    )

    parser.add_argument(
        "-scr",
        "--size_count_rel_threshold",
        type=float,
        help="Size threshold (relative) to count small segments",
        default=0.8,
    )

    parser.add_argument(
        "--output_basedir",
        help="Path to directory where results will be generated",
        default="./",
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    gtseg = Segmentation.from_file(args.gtseg)
    ossegs = [Segmentation.from_file(osseg) for osseg in args.ossegs]
    stats_df = Oversegmentation.compute_oversegmentation_statistics(
        gtseg,
        ossegs,
        args.segment_split_threshold,
        args.size_count_abs_threshold,
        args.size_count_rel_threshold,
    )
    stats_df.columns = args.ossegs
    print(stats_df)
    if args.output_basedir:
	if not os.path.exists(args.output_basedir):
		os.makedirs(path)
    output_fpath = os.path.join(args.output_basedir, 'stats.csv')
    stats_df.to_csv(output_fpath)
