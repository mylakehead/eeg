"""
Module Name: EEG Data Processing and Feature Extraction

Description:
    This module provides functions and utilities for processing EEG data, extracting features,
    and preparing the data for model training and evaluation. The functions cover a range of
    tasks including filtering raw EEG data, segmenting EEG signals into chunks, applying
    feature extraction methods, and organizing file paths for different EEG data sessions.

Functions:
    - filter_eeg: Applies a bandpass filter and common average referencing to raw EEG data.
    - split_eeg: Splits the EEG data into equal-length chunks based on a specified chunk length.
    - process_file: Processes a raw EEG data file, applies filtering and chunking, and assigns labels.
    - pad_and_split: Pads an array to a specified block size and splits it into blocks.
    - process_feature_file: Processes feature-based EEG data files according to the specified feature
      extraction method, chunking the data and assigning labels.
    - subject_file_map: Maps subjects and sessions to their corresponding EEG data files based on
      folder structure and file naming conventions.

Usage:
    This module is intended for preprocessing and feature extraction in EEG-based machine learning tasks.
    It supports multiple feature extraction methods and can handle different subjects and sessions in a
    structured way.

License:
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
    Created: 2024-10-16
    Last Modified: 2024-11-02
"""

import os
import re

import scipy
import numpy as np

from data.seed_iv import Subject, Session, FeatureMethod


def pad_and_split(array, block_size):
    """
    Pads an array to a specified block size and splits it into smaller blocks along the specified axis.

    :param array: The input 3D array to be padded and split, with shape (channels, length, features).
    :type array: np.ndarray
    :param block_size: The length of each block after splitting.
    :type block_size: int
    :return: A list of blocks, where each block has shape (channels, block_size, features).
    :rtype: list[np.ndarray]

    This function calculates the necessary padding to make the array length divisible by `block_size`.
    It pads the array by repeating the initial elements, if needed, and then splits the array into
    non-overlapping blocks of the specified `block_size`.

    Steps:
        1. Compute the padding length required to make the array length divisible by `block_size`.
        2. If padding is needed, repeat the initial portion of the array to fill the padding length.
        3. Split the padded array into smaller blocks along the length dimension.

    Example:
        Given an array with shape (channels, 100, features) and block_size of 20, the function pads
        the array to 120 if necessary, then splits it into 6 blocks of shape (channels, 20, features).
    """
    current_length = array.shape[1]

    pad_length = block_size - (current_length % block_size) if current_length % block_size != 0 else 0
    total_length = current_length + pad_length

    if pad_length > 0:
        pad_values = array[:, :pad_length, :]
        padded_array = np.concatenate((pad_values, array), axis=1)
    else:
        padded_array = array

    blocks = np.split(padded_array[:, :total_length, :], total_length // block_size, axis=1)
    return blocks


def process_feature_file_for_experiment_a(feature_method, file_name, labels, block_size):
    data = scipy.io.loadmat(file_name)
    data.pop('__header__', None)
    data.pop('__version__', None)
    data.pop('__globals__', None)

    train_chunk_list = list()
    train_label_list = list()

    test_chunk_list = list()
    test_label_list = list()

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

        if 1 <= trail_index <= 16:
            train_chunk_list.extend(chunks)
            train_label_list.extend([labels[label_index].value] * len(chunks))
        else:
            test_chunk_list.extend(chunks)
            test_label_list.extend([labels[label_index].value] * len(chunks))

        must.remove(trail_index)

    if len(must) != 0:
        raise Exception(f"file missing{must}")

    return train_chunk_list, train_label_list, test_chunk_list, test_label_list


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


def subject_file_map(folder: str) -> dict[Subject, dict[Session, str]]:
    """
    Creates a mapping of EEG data files by subject and session.

    :param folder: The path to the root folder containing the feature data organized by session.
    :type folder: str
    :return: A nested dictionary where each subject is mapped to another dictionary that maps each session
             to the corresponding file path.
    :rtype: dict[Subject, dict[Session, str]]

    This function scans the specified folder and organizes EEG data files into a nested dictionary structure
    based on subject and session. The expected folder structure is such that session folders contain files
    named with the format "<subject>_<date>.mat", where:
        - <subject> is the subject identifier (e.g., "1", "2").
        - <date> is an 8-digit date in the file name.

    Steps:
        1. Initialize an empty dictionary for each subject in the `file_map`.
        2. Traverse the folder structure recursively using `os.walk`.
        3. Extract the session number from the folder name and match files against a specific pattern.
        4. For each file matching the pattern, extract the subject ID and session number.
        5. Map each file path to the appropriate subject and session in `file_map`.

    Example:
        Given a folder structure like:
        ```
        root/
        ├── session1/
        │   ├── 1_20220101.mat
        │   ├── 2_20220101.mat
        ├── session2/
        │   ├── 1_20220102.mat
        │   ├── 2_20220102.mat
        ```

        `subject_file_map("root")` will return:
        ```
        {
            Subject.ONE: {Session.ONE: "root/session1/1_20220101.mat", Session.TWO: "root/session2/1_20220102.mat"},
            Subject.TWO: {Session.ONE: "root/session1/2_20220101.mat", Session.TWO: "root/session2/2_20220102.mat"},
        }
        ```

    Note:
        - The function assumes that the folder names represent session numbers, and the files follow the naming
          convention "<subject>_<date>.mat".
        - If a file does not match the expected pattern, it will be ignored.

    """
    file_map = dict()
    for s in Subject:
        file_map[s] = dict()

    pattern = r'(\d+)_(\d{8})\.mat'
    for root, _, files in os.walk(folder):
        paths = root.split(os.sep)
        session = paths[-1]

        for file_name in files:
            match = re.match(pattern, file_name)
            if match:
                s = int(match.group(1))
                # date
                _ = match.group(2)

                abs_path = os.path.join(root, file_name)
                file_map[Subject(s)][Session(int(session))] = abs_path

    return file_map
