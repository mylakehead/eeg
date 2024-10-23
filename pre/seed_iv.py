import os
import re

import scipy
import numpy as np

from data.seed_iv import Subject, Session, FeatureMethod


def filter_egg(raw, fs):
    # common average reference (CAR)
    average_reference = np.mean(raw, axis=0)
    car_eeg = raw - average_reference

    o = scipy.signal.butter(4, [0.15, 40], 'bandpass', fs=fs)
    filtered_eeg = scipy.signal.filtfilt(o[0], o[1], car_eeg, axis=1)

    return filtered_eeg


def split_eeg(eeg_data, chunk_length):
    channels, length = eeg_data.shape
    n_chunks = length // chunk_length

    chunks = np.array_split(eeg_data[:, :chunk_length * n_chunks].T, n_chunks)

    chunks_transposed = [chunk.T for chunk in chunks]

    return np.asarray(chunks_transposed)


def process_file(file_name, labels, fs=200, chunk_length=700):
    data = scipy.io.loadmat(file_name)
    data.pop('__header__', None)
    data.pop('__version__', None)
    data.pop('__globals__', None)

    chunk_list = list()
    label_list = list()

    pattern = r'\w+_eeg(\d+)'

    for index, trial in data.items():
        match = re.match(pattern, index)
        if not match:
            raise "\\w+_eeg(\\d+) pattern not match"
        label_index = int(match.group(1)) - 1

        filtered_egg = filter_egg(trial, fs)
        chunks = split_eeg(np.array(filtered_egg), chunk_length)

        chunk_list.extend(chunks)
        label_list += [labels[label_index].value] * chunks.shape[0]

    return chunk_list, label_list


def pad_and_split(array, block_size):
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


def process_feature_file(feature_method, file_name, labels, block_size):
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


def subject_file_map(folder):
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
