"""
Preprocessing of the EEG feature smooth data.

Copyright:
    MIT License

    Copyright © 2024 Lakehead University, Large Scale Data Analytics Group Project

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Authors:
    Kang Hong, XingJian Han, Minh Anh Nguyen
    hongkang@hongkang.name, xhan15@lakeheadu.ca, mnguyen9@lakeheadu.ca

Date:
    Created: 2024-10-02
    Last Modified: 2024-11-24
"""

import numpy as np

import scipy.io as sio

from data.seed_iv import subject_file_map, FeatureMethod, session_label


def pad_and_split(array, block_size, index, bands):
    """
    Pads an array along a specified dimension to make its size divisible by a block size,
    then splits the array into equally sized blocks along that dimension. Selects specific
    bands (indices) from the resulting blocks.

    Args:
        array (numpy.ndarray): Input array to be padded and split.
        block_size (int): Size of the blocks to split the array into.
        index (int): Dimension along which to pad and split the array (0, 1, or 2).
        bands (list): List of band objects containing `value` attributes (indices)
                      to select from each block.
    """
    current_length = array.shape[index]

    # Calculate the padding length to make the dimension divisible by block_size
    pad_length = block_size - (current_length % block_size) if current_length % block_size != 0 else 0
    total_length = current_length + pad_length

    if pad_length > 0:
        if index == 0:
            pad_values = array[:pad_length, :, :]
        elif index == 1:
            pad_values = array[:, :pad_length, :]
        elif index == 2:
            pad_values = array[:, :, :pad_length]
        else:
            raise IndexError
        padded_array = np.concatenate((pad_values, array), axis=index)
    else:
        padded_array = array

    # Split the padded array into equally sized blocks along the specified dimension
    if index == 0:
        blocks = np.split(padded_array[:total_length, :, :], total_length // block_size, axis=index)
    elif index == 1:
        blocks = np.split(padded_array[:, :total_length, :], total_length // block_size, axis=index)
    elif index == 2:
        blocks = np.split(padded_array[:, :, :total_length], total_length // block_size, axis=index)
    else:
        raise IndexError

    # Extract the specified bands (indices) from each block
    indexes = [band.value for band in bands]
    blocks = [block[indexes, :, :] for block in blocks]

    return blocks


def get_feature_dataset(data_path, subjects, sessions, trails, method, block_size, bands):
    """
    Generates a dataset of features and corresponding labels from EEG data files.

    Args:
        data_path (str): Path to the directory containing the EEG data files.
        subjects (list): List of subjects to include in the dataset.
        sessions (list): List of sessions to include for each subject.
        trails (list): List of trails to include for each session.
        method (FeatureMethod): The feature extraction method to apply (e.g., DE_LDS, PSD_LDS).
        block_size (int): The size of the blocks into which the data will be split.
        bands (list): List of band objects containing `value` attributes (indices) to extract.
    """
    dataset = []
    labels = []

    # Iterate over all subjects
    for subject in subjects:
        # Iterate over all sessions for the subject
        for session in sessions:
            subject_file_mapping = subject_file_map(data_path)
            files = subject_file_mapping[subject]
            file = files[session]

            data = sio.loadmat(file)
            data.pop('__header__', None)
            data.pop('__version__', None)
            data.pop('__globals__', None)

            for trail in trails:
                # Determine the data pattern based on the feature extraction method
                if method == FeatureMethod.DE_LDS:
                    pattern = f'de_LDS{trail+1}'
                elif method == FeatureMethod.DE_MOVING_AVE:
                    pattern = f'de_movingAve{trail+1}'
                elif method == FeatureMethod.PSD_LDS:
                    pattern = f'psd_LDS{trail+1}'
                elif method == FeatureMethod.PSD_MOVING_AVE:
                    pattern = f'psd_movingAve{trail+1}'
                else:
                    raise Exception("feature method error")

                found = False
                for key, trail_data in data.items():
                    if key != pattern:
                        continue

                    # Transpose the trail data to match the expected format
                    data_transpose = np.transpose(trail_data, (2, 1, 0))
                    chunks = pad_and_split(data_transpose, block_size, 1, bands)

                    dataset.extend(chunks)
                    labels.extend([session_label[session][trail].value] * len(chunks))

                    found = True

                if not found:
                    raise ModuleNotFoundError

    return np.array(dataset), np.array(labels)
