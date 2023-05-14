"""

Module that implements utility functions:

- volume reading wrapper to handle h5 and tiff volumes
- volume writing wrapper to handle h5 and tiff volumes
"""

import os
import h5py
import imageio
import pathlib
import numpy as np


def ensure_dir(dpath):
    head, tail = os.path.split(dpath)
    if not os.path.exists(dpath):
        if not os.path.exists(head):
            ensure_dir(head)
        os.mkdir(dpath)


def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def readh5(filename, dataset=None):
    fid = h5py.File(filename, "r")
    if dataset is None:
        dataset = list(fid)[0]
    return np.array(fid[dataset])


def readvol(filename, dataset=None):
    r"""Load a image volume in HDF5 or TIFF
    Code extracted from pytorch_connectomics framework
    """
    img_suf = filename[filename.rfind(".") + 1 :]
    if img_suf in ["h5", "hdf5"]:
        data = readh5(filename, dataset)
    elif "tif" in img_suf:
        data = imageio.volread(filename).squeeze()
    else:
        raise ValueError("unrecognizable file format for %s" % (filename))
    return data


def writevol(vol, fpath):
    fext = pathlib.Path(fpath).suffix
    if fext == ".tif":
        writeTIF(vol, fpath)
    elif fext == ".h5":
        writeh5(vol, fpath)


def writeh5(volume, fpath, datasetname="main"):
    fid = h5py.File(fpath, "w")
    if isinstance(datasetname, (list,)):
        for i, dd in enumerate(datasetname):
            ds = fid.create_dataset(
                dd, volume[i].shape, compression="gzip", dtype=volume[i].dtype
            )
            ds[:] = volume[i]
    else:
        ds = fid.create_dataset(
            datasetname, volume.shape, compression="gzip", dtype=volume.dtype
        )
        ds[:] = volume
    fid.close()


def writeTIF(volume, fpath):
    imageio.volwrite(fpath, volume)
