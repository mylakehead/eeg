import re
import numpy as np

import scipy.io as sio

from data.seed_iv import subject_file_map, FeatureMethod, session_label


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
    data = sio.loadmat(file_name)
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


def dataset_of_subject(folder: str, subject, feature_method, block_size: int) -> tuple[list[np.ndarray], list[int]]:
    subject_file_mapping = subject_file_map(folder)
    files = subject_file_mapping[subject]

    total_data_set = list()
    total_labels = list()
    for k, v in files.items():
        file = files[k]
        labels = session_label[k]
        data_set, data_labels = process_feature_file(feature_method, file, labels, block_size)

        total_data_set.extend(data_set)
        total_labels.extend(data_labels)

    return total_data_set, total_labels


def pad_and_split_new(array, block_size, index):
    current_length = array.shape[index]

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

    if index == 0:
        blocks = np.split(padded_array[:total_length, :, :], total_length // block_size, axis=index)
    elif index == 1:
        blocks = np.split(padded_array[:, :total_length, :], total_length // block_size, axis=index)
    elif index == 2:
        blocks = np.split(padded_array[:, :, :total_length], total_length // block_size, axis=index)
    else:
        raise IndexError

    # blocks = [block[1:2, :, :] for block in blocks]

    return blocks


def get_feature_dataset(data_path, subjects, sessions, trails, method, block_size):
    dataset = []
    labels = []

    for subject in subjects:
        for session in sessions:
            subject_file_mapping = subject_file_map(data_path)
            files = subject_file_mapping[subject]
            file = files[session]

            data = sio.loadmat(file)
            data.pop('__header__', None)
            data.pop('__version__', None)
            data.pop('__globals__', None)

            for trail in trails:
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

                    data_transpose = np.transpose(trail_data, (2, 1, 0))
                    chunks = pad_and_split_new(data_transpose, block_size, 1)

                    dataset.extend(chunks)
                    labels.extend([session_label[session][trail].value] * len(chunks))

                    found = True

                if not found:
                    raise ModuleNotFoundError

    return np.array(dataset), np.array(labels)
