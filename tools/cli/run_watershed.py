"""
Command-Line script used to run the watershed stage on affinity maps produced by the U-Net

It takes as parameters:

1. path to input affinity map
2. path to output directory
3. thresholds for zwatershed (2D segmentation)
4. thresholds for waterz (3D segmentation)

Then:

1. opens the input affinity map
2. apply subroutine in watershed.py module
3. store results in specified output path
"""

import os
import argparse
import numpy as np
import imageio
from lioness.utils import utils as Utils
from lioness.segmentation import watershed as Watershed


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("aff", help="Affinity graph")
    parser.add_argument("outdir", help="Output directory", type=str)

    parser.add_argument("--zw-thres", type=float, default=150.0)
    parser.add_argument("--zw-dust", type=float, default=150.0)
    parser.add_argument("--zw-low", type=float, default=0.15)
    parser.add_argument("--zw-high", type=float, default=0.9)
    parser.add_argument("--zw-dust-merge", type=float, default=0.2)
    parser.add_argument("--zw-mst-merge", type=float, default=0.7)
    parser.add_argument("--zw-rel", action="store_true")

    parser.add_argument("--wz-low", type=float, default=0.05)
    parser.add_argument("--wz-high", type=float, default=0.95)
    parser.add_argument(
        "--wz-thres", nargs="*", type=float, default=[0.1, 0.3, 0.5, 0.7]
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Process inference data to compute segmentation
    aff = Utils.readvol(args.aff)
    aff = (np.array(aff) / 255.0).astype(np.float32)
    segs = Watershed.run_watershed(
        aff,
        args.zw_thres,
        args.zw_dust,
        args.zw_low,
        args.zw_high,
        args.zw_dust_merge,
        args.zw_mst_merge,
        args.wz_thres,
        args.wz_low,
        args.wz_high,
        args.zw_rel,
    )

    # Extract inference result image file name
    head, tail = os.path.split(args.aff)
    fname, ext = os.path.splitext(tail)
    Utils.ensure_dir(args.outdir)

    # Save segmentations to files in the specified path
    for i, s in enumerate(segs):
        outfpath = os.path.join(
            args.outdir, fname + "-seg" + ("%.2f" % args.wz_thres[i]) + ".h5"
        )
        Utils.writevol(s, outfpath)
        outfpath = os.path.join(
            args.outdir, fname + "-seg" + ("%.2f" % args.wz_thres[i]) + ".tif"
        )
        Utils.writevol(s, outfpath)
