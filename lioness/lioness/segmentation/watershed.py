"""

Module that implements watershed functionality as a two stage process where

1. watershed is performed per-slice, in a 2D domain, obtaining the fragments mask
2. watershed is performed at a volume level, in a 3D domain, and uses the previously generated fragments mask
"""

import numpy as np
import waterz
import zwatershed
from lioness.segmentation import segmentation as Segmentation


def run_watershed(
    aff,
    zw_thres,
    zw_dust,
    zw_low,
    zw_high,
    zw_dust_merge,
    zw_mst_merge,
    wz_thres,
    wz_low,
    wz_high,
    zw_rel=False,
):
    # Run zwatershed
    volume_size = list(aff.shape[1:])
    zw2d = np.zeros(volume_size, np.uint32)
    max_id = np.uint32(0)
    for zi in range(volume_size[0]):
        aff_z = aff[:, zi : zi + 1]
        seg = zwatershed.zwatershed(
            aff_z,
            T_threshes=zw_thres if isinstance(zw_thres, list) else [zw_thres],
            T_dust=zw_dust,
            T_aff=[zw_low, zw_high, zw_dust_merge],
            T_aff_relative=zw_rel,
            T_merge=zw_mst_merge,
        )[0][0][0]
        seg[seg > 0] += max_id
        max_id = seg.max()
        zw2d[zi] = Segmentation.relabel(seg)

    # Run waterz
    zw2d_nonoverlap = np.zeros(zw2d.shape, np.uint64)
    max_id = np.uint64(0)
    for zi in range(zw2d.shape[0]):
        seg = np.array(zw2d[zi])
        seg[seg > 0] += max_id
        max_id = seg.max()
        zw2d_nonoverlap[zi] = seg

    print("do wz3d")
    segs = waterz.waterz(
        aff,
        wz_thres if isinstance(wz_thres, list) else [wz_thres],
        merge_function="aff50_his256",
        aff_threshold=[wz_low, wz_high],
        fragments_mask=zw2d_nonoverlap,
    )

    for seg_id in range(len(segs)):
        segs[seg_id] = Segmentation.relabel(segs[seg_id], do_dtype=True)
    return segs
