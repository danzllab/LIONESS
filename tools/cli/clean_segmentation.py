"""
Command-Line script used to clean segmentations produced by the automatic segmentation pipeline.

It takes as parameters:

1. the minimum segment size
2. the minimum number of segment slices
3. boolean indicating if relabeling is requested
4. path to output segmentation

Then:

1. it removes all segments that are smaller than the specified size
2. it removes all segments that contain less that the specified number of slices in any axis
3. if requested, segments are relabelled. This may help reduce file size if the total number of
   segments is reduced below the 8-bit or 16-bit range, e.g. if the number of segments decreases
   from larger than 65535 (32-bit data type) to smaller than 65536 (16-bit data type)
4. print segmentation information as output
"""

import argparse
import pandas as pd
import lioness.segmentation.segmentation as Segmentation


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("seg", help="Path to segmentation")
    parser.add_argument("--size", help="Specify minimum segment size", type=int)
    parser.add_argument("--slices", help="Specify minimum segment slices", type=int)
    parser.add_argument("--relabel", help="Request relabel", action="store_true")
    parser.add_argument("--output", help="Specify output filename")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    seg = Segmentation.from_file(args.seg)
    df = seg.get_info_dataframe()
    print("Cleaning %s..." % args.seg)
    print("Before:")
    print(df)
    if args.size is not None:
        seg.remove_small_segments(args.size)
    if args.slices is not None:
        seg.remove_few_slice_segments(args.slices)
    if args.relabel:
        seg.relabel(do_dtype=True)
    df = seg.get_info_dataframe()
    print("After:")
    print(df)
    if args.output is not None:
        seg.save(args.output)
