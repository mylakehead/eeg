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









