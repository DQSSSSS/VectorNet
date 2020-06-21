import os
import shutil
import tempfile
import uuid
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster.dbscan_ import DBSCAN

import h5py

TYPE_LIST = Union[List[np.ndarray], np.ndarray]


def generate_forecasting_h5(
    data: Dict[int, TYPE_LIST],
    output_path: str,
    filename: str = "argoverse_forecasting_baseline",
    probabilities: Optional[Dict[int, List[float]]] = None,
) -> None:
    """
    Helper function to generate the result h5 file for argoverse forecasting challenge

    Args:
        data: a dictionary of trajectory, with the key being the sequence ID. For each sequence, the
              trajectory should be stored in a (9,30,2) np.ndarray
        output_path: path to the output directory to store the output h5 file
        filename: to be used as the name of the file
        probabilities (optional) : normalized probability for each trajectory

    Returns:

    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hf = h5py.File(os.path.join(output_path, filename + ".h5"), "w")
    future_frames = 30
    d_all: List[np.ndarray] = []
    counter = 0
    for key, value in data.items():
        print("\r" + str(counter + 1) + "/" + str(len(data)), end="")

        if isinstance(value, List):
            value = np.array(value)
        assert value.shape[1:3] == (
            future_frames,
            2,
        ), f"ERROR: the data should be of shape (n,30,2), currently getting {value.shape}"

        n = value.shape[0]
        len_val = len(value)
        value = value.reshape(n * future_frames, 2)
        if probabilities is not None:
            assert key in probabilities.keys(), f"missing probabilities for sequence {key}"
            assert (
                len(probabilities[key]) == len_val
            ), f"mismatch sequence and probabilities len for {key}: {len(probabilities[key])} !== {len_val}"
            # assert np.isclose(np.sum(probabilities[key]), 1), "probabilities are not normalized"

            d = np.array(
                [
                    [key, np.float32(x), np.float32(y), probabilities[key][int(np.floor(i / future_frames))]]
                    for i, (x, y) in enumerate(value)
                ]
            )
        else:
            d = np.array([[key, np.float32(x), np.float32(y)] for x, y in value])

        if len(d_all) == 0:
            d_all = d
        else:
            d_all = np.concatenate([d_all, d], 0)
        counter += 1

    hf.create_dataset("argoverse_forecasting", data=d_all, compression="gzip", compression_opts=9)
    hf.close()
