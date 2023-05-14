"""
Command-Line script used to preprocess input datasets

It takes as parameters:

1. path to input dataset
2. path to output dataset

Then:

1. opens the input dataset
2. stretches intensity values to span the [0,1] range
3. converts to 8-bit format
4. stores output dataset in specified path
"""

import h5py
import imageio
import argparse
import numpy as np
import pathlib
from lioness.utils import utils as Utils

parser = argparse.ArgumentParser()
parser.add_argument("img_fname", help="Specify input image file")
parser.add_argument("output", help="Specify output filename")
args = parser.parse_args()


print(args.img_fname)
image = Utils.readvol(args.img_fname)
print("Source Image:", image.shape, image.dtype, image.min(), image.max())
normalized = (image - image.min()) / float(image.max() - image.min())
print(
    "Normalized Image:",
    normalized.shape,
    normalized.dtype,
    normalized.min(),
    normalized.max(),
)
normalized = (normalized * 255).astype(np.uint8)
print(
    "8-bit Image:",
    normalized.shape,
    normalized.dtype,
    normalized.min(),
    normalized.max(),
)
p = pathlib.Path(args.output)
out_ext = p.suffix.lower()
print("Generating", args.output)
Utils.writevol(normalized, args.output)
