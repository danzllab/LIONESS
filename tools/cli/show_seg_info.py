"""
Command-Line script used to print segmentation information:

- number of segments"nsegments
- average segment size
- minimum segment size
- maximum segment size
- standard deviation in segment size
- average number of segment slices
- minimum number of segment slices
- maximum number of segment slices
- data type
- volume size

It takes as parameters:

1. path to input segmentation

Then:

1. opens the input segmentation
2. calls subroutine in segmentation.py module
3. prints segmentation info
"""

import argparse
import numpy as np
import pandas as pd
from lioness.segmentation import segmentation as Segmentation


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("seg", help="Path to segmentation file")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    seg = Segmentation.from_file(args.seg)
    df = seg.get_info_dataframe()
    print(df)
