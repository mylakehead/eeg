import os
import re

import scipy
import numpy as np
import tensorflow as tf

from data.seed_iv import Subject, Session, session_label


def get_subject_file_mapping(folder):
    subject_files = dict()
    for s in Subject:
        subject_files[s] = dict()

    pattern = r'(\d+)_(\d{8})\.mat'
    for root, _, files in os.walk(folder):
        paths = root.split('/')
        session = paths[-1]

        for file_name in files:
            match = re.match(pattern, file_name)
            if match:
                s = int(match.group(1))
                # date
                _ = match.group(2)

                abs_path = os.path.join(root, file_name)
                subject_files[Subject(s)][Session(int(session))] = abs_path

    return subject_files


def reshape_chunk(chunk_list):
    n = chunk_list[0]

    for trial in chunk_list[1:]:
        n = np.vstack([n, trial])

    n = n.transpose((0, 2, 1))
    n = n.reshape(n.shape[0], n.shape[1], n.shape[2], 1)

    return n


def pre_process(folder, subject):
    subject_file_mapping = get_subject_file_mapping(folder)
    files = subject_file_mapping[subject]

    s1_train_set, s1_train_label = process_file(files[Session.ONE], session_label[Session.ONE])
    s2_train_set, s2_train_label, s2_test_set, s2_test_label = split_train_test(
        files[Session.TWO], session_label[Session.TWO]
    )
    s3_train_set, s3_train_label, s3_test_set, s3_test_label = split_train_test(
        files[Session.THREE], session_label[Session.THREE]
    )

    s1_train = reshape_chunk(s1_train_set)
    s2_train = reshape_chunk(s2_train_set)
    s2_test = reshape_chunk(s2_test_set)
    s3_train = reshape_chunk(s3_train_set)
    s3_test = reshape_chunk(s3_test_set)

    c_s1_train_label = tf.keras.utils.to_categorical(s1_train_label)
    c_s2_train_label = tf.keras.utils.to_categorical(s2_train_label)
    c_s2_test_label = tf.keras.utils.to_categorical(s2_test_label)
    c_s3_train_label = tf.keras.utils.to_categorical(s3_train_label)
    c_s3_test_label = tf.keras.utils.to_categorical(s3_test_label)

    return (
        s1_train, s2_train, s2_test, s3_train, s3_test,
        c_s1_train_label, c_s2_train_label, c_s2_test_label, c_s3_train_label, c_s3_test_label
    )


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

    return np.asarray(chunks)


def split_train_test(file_name, labels):
    data = scipy.io.loadmat(file_name)
    data.pop('__header__', None)
    data.pop('__version__', None)
    data.pop('__globals__', None)

    half = len(data.items()) // 2

    test_chunk_list = []
    test_label_list = []
    train_chunk_list = []
    train_label_list = []

    pattern = r'hql_eeg(\d+)'

    for index, trial in data.items():
        match = re.match(pattern, index)
        if not match:
            raise "hql_eeg pattern not match"

        label_index = int(match.group(1)) - 1

        if label_index >= half:
            # TODO fs, chunk configuration
            filtered_egg = filter_egg(trial, 200)
            chunks = split_eeg(np.array(filtered_egg), 700)

            test_chunk_list.append(chunks)
            test_label_list += [labels[label_index].value] * chunks.shape[0]
        else:
            # TODO fs, chunk configuration
            filtered_egg = filter_egg(trial, 200)
            chunks = split_eeg(np.array(filtered_egg), 700)

            train_chunk_list.append(chunks)
            train_label_list += [labels[label_index].value] * chunks.shape[0]

    return train_chunk_list, train_label_list, test_chunk_list, test_label_list


def process_file(file_name, labels):
    data = scipy.io.loadmat(file_name)
    data.pop('__header__', None)
    data.pop('__version__', None)
    data.pop('__globals__', None)

    chunk_list = list()
    label_list = list()

    pattern = r'hql_eeg(\d+)'

    for index, trial in data.items():
        match = re.match(pattern, index)
        if not match:
            raise "hql_eeg pattern not match"
        label_index = int(match.group(1)) - 1

        # TODO fs, chunk configuration
        filtered_egg = filter_egg(trial, 200)
        chunks = split_eeg(np.array(filtered_egg), 700)

        chunk_list.append(chunks)
        label_list += [labels[label_index].value] * chunks.shape[0]

    return chunk_list, label_list


if __name__ == '__main__':
    split_train_test(
        '/Users/hongkang/github/mylakehead/eeg/.dataset/SEED_IV/eeg_raw_data/2/3_20151018.mat',
        session_label[Session.TWO]
    )
