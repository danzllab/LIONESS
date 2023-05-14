"""

Module the implements functionality to compute oversegmentation metrics given
an input segmentation and a ground truth segmentation

- number of small segments in other segmentation
- ratio of number of segments in ground truth vs number of segments in other segmentation
- segment split: number of segments in other segmentation that overlap a given segmentation in ground truth
"""


import numpy as np
import pandas as pd
import scipy.sparse as sparse
from lioness.segmentation import segmentation as Segmentation


def compute_overlay_matrix(gtseg, osseg):
    return gtseg.overlay_matrix(osseg)


def compute_segment_split_dataframe(overlay_df, gtdf, min_overlap=0):
    df = (
        overlay_df[overlay_df.ratio_os > min_overlap]
        .groupby(["sid_gt"])["sid_os"]
        .count()
        .reset_index()
        .rename(columns={"sid_gt": "sid", "sid_os": "split_count"})
    )
    df = pd.merge(df, gtdf[["sid", "size"]], on="sid").sort_values(by="split_count")
    return df


def compute_overlay_dataframe(
    gtseg, osseg, gtdf=None, osdf=None, overlay_matrix=None, drop_background=True
):
    if gtdf is None:
        gtdf = gtseg.segment_size_dataframe()
    if osdf is None:
        osdf = osseg.segment_size_dataframe()
    if overlay_matrix is None:
        overlay_matrix = compute_overlay_matrix(gtseg, osseg)
    values = overlay_matrix.data
    rows = overlay_matrix.row
    cols = overlay_matrix.col
    overlay_df = pd.DataFrame({"sid_gt": rows, "sid_os": cols, "size": values})
    if drop_background:
        overlay_df = overlay_df[(overlay_df.sid_gt != 0) & (overlay_df.sid_os != 0)]
    overlay_df = pd.merge(
        overlay_df,
        gtdf,
        how="left",
        left_on="sid_gt",
        right_on="sid",
        suffixes=["", "_gt"],
    ).drop(["sid"], axis=1)
    overlay_df = pd.merge(
        overlay_df,
        osdf,
        how="left",
        left_on="sid_os",
        right_on="sid",
        suffixes=["", "_os"],
    ).drop(["sid"], axis=1)
    overlay_df["ratio_gt"] = overlay_df["size"] / overlay_df["size_gt"]
    overlay_df["ratio_os"] = overlay_df["size"] / overlay_df["size_os"]
    return overlay_df


def compute_oversegmentation_statistics(
    gtseg,
    ossegs,
    segment_split_threshold,
    abs_segment_size_threshold,
    rel_segment_size_threshold,
):
    gtdf = gtseg.segment_size_dataframe(drop_background=False)
    nsegs = len(ossegs)

    osdf_list = [seg.segment_size_dataframe(drop_background=False) for seg in ossegs]
    overlay_matrix_list = [compute_overlay_matrix(gtseg, osseg) for osseg in ossegs]
    overlay_df_list = [
        compute_overlay_dataframe(
            gtseg,
            ossegs[i],
            gtdf=gtdf,
            osdf=osdf_list[i],
            overlay_matrix=overlay_matrix_list[i],
        )
        for i in range(nsegs)
    ]

    stats_dict = {}
    n_segments_gt_list = [len(gtdf)] * len(osdf_list)
    stats_dict["N_segments_gt"] = n_segments_gt_list
    n_segments_as_list = [len(osdf) for osdf in osdf_list]
    stats_dict["N_segments_os"] = n_segments_as_list
    n_segments_ratio_list = [
        float(n_segments_as_list[i]) / n_segments_gt_list[i] for i in range(nsegs)
    ]
    stats_dict["N_segments_ratio"] = n_segments_ratio_list
    segment_split_dfs = [
        compute_segment_split_dataframe(
            overlay_df_list[i], gtdf, segment_split_threshold
        )
        for i in range(len(osdf_list))
    ]
    max_split_count_list = [
        segment_split_dfs[i].split_count.max() for i in range(nsegs)
    ]
    stats_dict["max_split_count"] = max_split_count_list
    mean_split_count_list = [
        segment_split_dfs[i].split_count.mean() for i in range(nsegs)
    ]
    stats_dict["mean_split_count"] = mean_split_count_list

    nsegs_list = [
        float(len(dfas[dfas["size"] < abs_segment_size_threshold])) / len(dfas)
        for dfas in osdf_list
    ]
    stats_dict["N_small_segments_abs"] = nsegs_list

    nsegs_list = [
        float(len(dfas[dfas["size"] < rel_segment_size_threshold * gtdf.size.min()]))
        / len(dfas)
        for dfas in osdf_list
    ]
    stats_dict["N_small_segments_rel"] = nsegs_list

    stats_df = pd.DataFrame.from_dict(stats_dict).transpose()
    return stats_df
