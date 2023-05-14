"""

Module that implements functionality regarding segmentations

- open segmentations from input file path
- relabel segmentations (adapting data type if requested)
- compute segmentation statistics such as
  - number of segments
  - segment size
  - segment slices
- compute segmentation ARI error
- compute segmentation statistics comparing two segmentations
  - overlay matrix between two segmentations where each row indicates
    a segment ID in the reference segmentation and each column a segment ID
    in the other segmentation (used to compute segment split)
- remove segments smaller than a given value
- remove segments composed of fewer slices than a given value
- resample segmentations using ITK's resampling methods
"""

import numpy as np
import scipy.sparse as sparse
import pandas as pd
import itk
from lioness.utils import utils as Utils


class Segmentation:
    def __init__(self, data):
        self.data = data
        assert isinstance(self.data, np.ndarray), "ERROR: expected numpy array"

    def nsegments(self):
        """Return number of segments in the segmentation

        Returns:
            int: number of unique segment IDs
        """
        usids = self.segment_IDs()
        if 0 in usids:
            usids = usids[1:]
        return len(usids)

    def segment_IDs(self):
        return np.unique(self.data)

    def save(self, fpath):
        Utils.writevol(self.data, fpath)

    def dtype(self):
        """Return type of data in the segmentation (int16, uint32...)

        Returns:
            dtype: type of data
        """
        return self.data.dtype

    def relabelDtype(self):
        """Relabel data type of segmentation data"""
        max_id = self.data.max()
        m_type = np.uint64
        if max_id < 2 ** 8:
            m_type = np.uint8
        elif max_id < 2 ** 16:
            m_type = np.uint16
        elif max_id < 2 ** 32:
            m_type = np.uint32
        self.data = self.data.astype(m_type)

    def relabel(self, do_dtype=False):
        """Relabel so that semgent IDs are consecutive

        Args:
            do_dtype (bool, optional): Indicate if dtype relabeling should be performed too. Defaults to False.
        """
        if self.data is None or self.data.max() == 0:
            return
        uid = np.unique(self.data)
        uid = uid[uid > 0]
        max_id = int(max(uid))
        mapping = np.zeros(max_id + 1, dtype=self.data.dtype)
        mapping[uid] = np.arange(1, len(uid) + 1)
        self.data = mapping[self.data]
        if do_dtype:
            self.relabelDtype()

    def get_info_dataframe(self):
        """Create and return a dataframe containing information about the segmentation

        Returns:
            pandas dataframe: dataframe containing the information of the segmentation
        """
        usids, counts = np.unique(self.data, return_counts=True)
        if 0 in usids:
            usids = usids[1:]
            counts = counts[1:]
        maxsize = np.max(counts)
        minsize = np.min(counts)
        avgsize = np.mean(counts)
        stdsize = np.std(counts)
        nsegments = len(usids)
        size = self.data.shape
        _, slice_count = self.count_segment_slices(drop_background=True)
        minslices = np.min(slice_count)
        maxslices = np.max(slice_count)
        avgslices = np.mean(slice_count)
        data = {
            "nsegments": [nsegments],
            "avgsize": [avgsize],
            "minsize": [minsize],
            "maxsize": [maxsize],
            "stdsize": [stdsize],
            "avgslices": [avgslices],
            "minslices": [minslices],
            "maxslices": [maxslices],
            "dtype": [self.data.dtype],
            "size": [size],
        }
        df = pd.DataFrame.from_dict(data, orient="index")
        return df

    def adapted_rand(self, gt, all_stats=False):
        """Compute Adapted Rand error

        Code borrowed from pytorch_connectomics from Harvard's VCG group

        Formula is given as 1 - the maximal F-score of the Rand index
        (excluding the zero component of the original labels). Adapted
        from the SNEMI3D MATLAB script, hence the strange style.

        Args:
            gt (np.ndarray): the groundtruth to score against, where each value is a label. same shape as seg
            all_stats (bool, optional): whether to also return precision and recall as a 3-tuple with rand_error. Defaults to False.

        Returns:
            are (float): The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$, where $p$ and $r$ are the precision and recall described below.
            prec (float, optional): The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
            rec (float, optional): The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)

        References:
        [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
        """
        # segA is truth, segB is query
        segA = np.ravel(gt)
        segB = np.ravel(self.data)
        n = segA.size

        n_labels_A = np.amax(segA) + 1
        n_labels_B = np.amax(segB) + 1

        ones_data = np.ones(n, int)

        p_ij = sparse.csr_matrix(
            (ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B)
        )

        a = p_ij[1:n_labels_A, :]
        b = p_ij[1:n_labels_A, 1:n_labels_B]
        c = p_ij[1:n_labels_A, 0].todense()
        d = b.multiply(b)

        a_i = np.array(a.sum(1))
        b_i = np.array(b.sum(0))

        sumA = np.sum(a_i * a_i)
        sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
        sumAB = np.sum(d) + (np.sum(c) / n)

        precision = np.float32(sumAB) / sumB
        recall = np.float32(sumAB) / sumA

        fScore = 2.0 * precision * recall / (precision + recall)
        are = 1.0 - fScore

        if all_stats:
            return (are, precision, recall)
        else:
            return are

    def segment_size_dataframe(self, drop_background=True):
        sids, counts = np.unique(self.data, return_counts=True)
        df = pd.DataFrame({"sid": sids, "size": counts})
        if drop_background:
            df = df[df["sid"] != 0]
        return df

    def overlay_matrix(self, seg):
        segA = np.ravel(self.data)
        segB = np.ravel(seg.data)
        n = segA.size

        nlabelsA = np.amax(segA) + 1
        nlabelsB = np.amax(segB) + 1

        ones_data = np.ones(n, int)

        p_ij = sparse.csr_matrix(
            (ones_data, (segA[:], segB[:])), shape=(nlabelsA, nlabelsB)
        )
        p_ij = p_ij.tocoo()
        return p_ij

    def count_segment_slices_on_axis(self, axis, drop_background=True):
        uids = np.unique(self.data)
        if axis == 0:
            auids = [np.unique(self.data[i, :, :]) for i in range(self.data.shape[0])]
        elif axis == 1:
            auids = [np.unique(self.data[:, i, :]) for i in range(self.data.shape[1])]
        elif axis == 2:
            auids = [np.unique(self.data[:, :, i]) for i in range(self.data.shape[2])]
        count = np.zeros(np.max(uids) + 1)
        for i in range(self.data.shape[axis]):
            count[auids[i]] += 1
        return count

    def count_segment_slices(self, drop_background=True):
        """Count segment slices

        Args:
            drop_background (bool, optional): Indicate if background segment should be considered

        Returns:
            uids (1D ndarray): segment IDs
            slice_count (2D ndarray): slice count for each segment ID and axis (Nx3)
        """
        count0 = self.count_segment_slices_on_axis(axis=0)
        count1 = self.count_segment_slices_on_axis(axis=1)
        count2 = self.count_segment_slices_on_axis(axis=2)
        uids = np.unique(self.data)
        slice_count = np.vstack([count0, count1, count2]).transpose()
        slice_count = slice_count[uids, :]
        if drop_background:
            slice_count = slice_count[uids != 0, :]
            uids = uids[uids != 0]
        return uids, slice_count

    def remove_small_segments(self, nminvoxels):
        usids, count = np.unique(self.data, return_counts=True)
        small_usids = usids[count <= nminvoxels]
        print("Found %d small (size <= %d) segments:" % (len(small_usids), nminvoxels))
        mask = np.isin(self.data, small_usids)
        self.data[mask] = 0

    def remove_few_slice_segments(self, nminslices):
        usids, slice_count = self.count_segment_slices(drop_background=True)
        min_slice_count = np.min(slice_count, axis=1)
        few_slice_segments = usids[min_slice_count <= nminslices]
        print(
            "Found %d few slice (nslices <= %d) segments:"
            % (len(few_slice_segments), nminslices)
        )
        mask = np.isin(self.data, few_slice_segments)
        self.data[mask] = 0

    def remove_segments(self, sids):
        for sid in sids:
            self.data = np.where(self.data == sid, 0, self.data)

    def resample(self, resample_ratio, interpolator_str="gaussian"):
        itk_data = itk.GetImageFromArray(self.data)
        input_size = itk.size(itk_data)
        print("input_size,", input_size)
        input_spacing = itk.spacing(itk_data)
        print("input_spacing,", input_spacing)
        input_origin = itk.origin(itk_data)
        print("input_origin,", input_origin)
        Dimension = itk_data.GetImageDimension()

        output_size = [int(input_size[d] * resample_ratio[d]) for d in range(Dimension)]
        print("output_size,", output_size)
        output_spacing = [
            input_spacing[d] / resample_ratio[d] for d in range(Dimension)
        ]
        print("output_spacing,", output_spacing)
        output_origin = [
            input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
            for d in range(Dimension)
        ]
        print("output_origin,", output_origin)

        if interpolator_str == "gaussian":
            interpolator = itk.LabelImageGaussianInterpolateImageFunction.New(itk_data)
        elif interpolator_str == "nearest_neighbor":
            interpolator = itk.NearestNeighborInterpolateImageFunction.New(itk_data)
        else:
            raise ValueError("Unknown interpolator")
        print("Using interpolator:", interpolator_str)

        resampled = itk.resample_image_filter(
            self.data,
            interpolator=interpolator,
            size=output_size,
            output_spacing=output_spacing,
            output_origin=output_origin,
        )
        return Segmentation(np.array(resampled))


def from_file(fpath_str):
    """Load segmentation from file

    Args:
        fpath_str (str): String indicating the path to the file containing the segmentation

    Returns:
        Segmentation: Segmentation object containing the volume
    """
    seg = Utils.readvol(fpath_str)
    print("Loaded segmentation %s" % fpath_str)
    return Segmentation(seg)


def relabelDtype(seg):
    """Relabel data type of segmentation data"""
    max_id = seg.max()
    m_type = np.uint64
    if max_id < 2 ** 8:
        m_type = np.uint8
    elif max_id < 2 ** 16:
        m_type = np.uint16
    elif max_id < 2 ** 32:
        m_type = np.uint32
    return seg.astype(m_type)


def relabel(seg, do_dtype=False):
    """Relabel so that semgent IDs are consecutive

    Args:
        do_dtype (bool, optional): Indicate if dtype relabeling should be performed too. Defaults to False.
    """
    if seg is None or seg.max() == 0:
        return seg
    uid = np.unique(seg)
    uid = uid[uid > 0]
    max_id = int(max(uid))
    mapping = np.zeros(max_id + 1, dtype=seg.dtype)
    mapping[uid] = np.arange(1, len(uid) + 1)
    if do_dtype:
        return relabelDtype(mapping[seg])
    else:
        return mapping[seg]


def _test_init():
    maxsid = 10
    data = np.random.randint(0, maxsid, size=(10, 10, 10))
    seg = Segmentation(data)
    assert (seg.data == data).all(), "[test_init]...ERROR"


def _test_nsegments_with_zero():
    maxsid = 10
    data = np.random.randint(0, maxsid, size=(10, 10, 10))
    seg = Segmentation(data)
    assert seg.nsegments() == (maxsid - 1), "[test_nsegments_with_zero]...ERROR"


def _test_nsegments_without_zero():
    maxsid = 10
    data = np.random.randint(1, maxsid, size=(10, 10, 10))
    seg = Segmentation(data)
    assert seg.nsegments() == (maxsid - 1), "[test_nsegments_without_zero]...ERROR"


def _test_dtype():
    maxsid = 10
    data = np.random.randint(0, maxsid, size=(10, 10, 10), dtype=np.uint16)
    seg = Segmentation(data)
    assert seg.dtype() == np.uint16, "[test_dtype]...ERROR"


def _test_get_info_dataframe():
    sids = [0, 1, 2, 3, 4, 5]
    ssizes = [400, 250, 150, 100, 50, 50]
    data = np.hstack(
        [sids[i] * np.ones((ssizes[i],), dtype=np.uint16) for i in range(len(sids))]
    )
    np.random.shuffle(data)
    data = data.reshape((10, 10, 10))
    seg = Segmentation(data)
    df = seg.get_info_dataframe()
    assert df.loc["nsegments"][0] == len(sids) - 1
    assert df.loc["avgsize"][0] == np.mean(ssizes[1:])
    assert df.loc["minsize"][0] == 50
    assert df.loc["maxsize"][0] == 250
    assert df.loc["stdsize"][0] == np.std(ssizes[1:])


def _test_relabelDtype():
    vol8 = np.random.randint(0, 2 ** 8, size=(10, 10, 10), dtype=np.uint64)
    vol16 = np.random.randint(0, 2 ** 16, size=(10, 10, 10), dtype=np.uint64)
    vol32 = np.random.randint(0, 2 ** 32, size=(10, 10, 10), dtype=np.uint64)
    vol8t = Segmentation.relabelDtype(vol8)
    vol16t = Segmentation.relabelDtype(vol16)
    vol32t = Segmentation.relabelDtype(vol32)
    assert vol8t.dtype == np.uint8, "ERROR: wrong bit depth for 8-bit volume "
    assert vol16t.dtype == np.uint16, "ERROR: wrong bit depth for 16-bit volume "
    assert vol32t.dtype == np.uint32, "ERROR: wrong bit depth for 32-bit volume "


def _test_relabel():
    vol8 = np.random.randint(1, 2 ** 8, size=(4, 4, 4), dtype=np.uint32)
    vol16 = np.random.randint(1, 2 ** 16, size=(4, 4, 4), dtype=np.uint32)
    vol32 = np.random.randint(1, 2 ** 32, size=(4, 4, 4), dtype=np.uint32)
    vol8d = Segmentation.relabel(vol8, do_dtype=False)
    vol16d = Segmentation.relabel(vol16, do_dtype=False)
    vol32d = Segmentation.relabel(vol32, do_dtype=False)
    assert len(np.unique(vol8d)) == np.max(
        vol8d
    ), "ERROR: (vol8d) max label (%d) does not match number of labels (%d)" % (
        np.max(vol8d),
        len(np.unique(vol8d)),
    )
    assert len(np.unique(vol16d)) == np.max(
        vol16d
    ), "ERROR: (vol16d) max label (%d) does not match number of labels (%d)" % (
        np.max(vol16d),
        len(np.unique(vol16d)),
    )
    assert len(np.unique(vol32d)) == np.max(
        vol32d
    ), "ERROR: (vol32d) max label (%d) does not match number of labels (%d)" % (
        np.max(vol32d),
        len(np.unique(vol32d)),
    )


def run_tests():
    _test_init()
    _test_nsegments_with_zero()
    _test_nsegments_without_zero()
    _test_dtype()
    _test_get_info_dataframe()
    _test_relabel()
    _test_relabelDtype()


if __name__ == "__main__":
    run_tests()
