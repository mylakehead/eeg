import os
import re

import scipy
import numpy as np

from data.seed_iv import Subject, Session, FeatureMethod






def process_feature_file(feature_method, file_name, labels, block_size) -> tuple[list[np.ndarray], list[int]]:
    """
    Processes a feature-based EEG data file according to the specified feature extraction method,
    splits the data into chunks of a specified block size, and assigns labels to each chunk.

    :param feature_method: The feature extraction method to use, such as DE_LDS, DE_MOVING_AVE, PSD_LDS,
    or PSD_MOVING_AVE.
    :type feature_method: FeatureMethod
    :param file_name: The path to the .mat file containing the EEG feature data.
    :type file_name: str
    :param labels: A list of labels corresponding to each trial in the data file.
    :param block_size: The block size to split each trial into non-overlapping segments.
    :type block_size: int
    :return: A tuple containing two lists:
        - chunk_list: A list of np.ndarray blocks, each representing a chunk of EEG data.
        - label_list: A list of integer labels corresponding to each chunk in `chunk_list`.
    :rtype: tuple[list[np.ndarray], list[int]]

    This function performs the following steps:
        1. Loads the .mat file and removes unnecessary metadata fields.
        2. Determines the correct regex pattern based on the specified feature extraction method.
        3. Iterates through each trial in the data file:
            - Matches the trial name using the regex pattern to identify trial indices.
            - Transposes the trial data to ensure correct shape.
            - Splits the trial data into chunks of size `block_size` using `pad_and_split`.
            - Appends each chunk to `chunk_list` and assigns the corresponding label from `labels`.
        4. Ensures that all expected trials (1 through 24) are present in the file. Raises an exception
           if any trial is missing.

    Raises:
        Exception: If an invalid feature method is specified, or if any expected trials are missing from the file.

    """
    data = scipy.io.loadmat(file_name)
    data.pop('__header__', None)
    data.pop('__version__', None)
    data.pop('__globals__', None)

    chunk_list = list()
    label_list = list()

    if feature_method == FeatureMethod.DE_LDS:
        pattern = r'de_LDS(\d+)'
    elif feature_method == FeatureMethod.DE_MOVING_AVE:
        pattern = r'de_movingAve(\d+)'
    elif feature_method == FeatureMethod.PSD_LDS:
        pattern = r'psd_LDS(\d+)'
    elif feature_method == FeatureMethod.PSD_MOVING_AVE:
        pattern = r'psd_movingAve(\d+)'
    else:
        raise Exception("feature method error")

    must = list(range(1, 25))
    for index, trial in data.items():
        match = re.match(pattern, index)
        if not match:
            continue

        trail_index = int(match.group(1))
        label_index = trail_index - 1

        trail_transpose = np.transpose(trial, (2, 1, 0))

        chunks = pad_and_split(trail_transpose, block_size)

        chunk_list.extend(chunks)
        label_list.extend([labels[label_index].value] * len(chunks))

        must.remove(trail_index)

    if len(must) != 0:
        raise Exception(f"file missing{must}")

    return chunk_list, label_list



