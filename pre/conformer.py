import numpy as np

import scipy.io as sio

from data.seed_iv import subject_file_map, FeatureMethod, session_label


def get_raw_dataset(data_path, subjects, sessions, trails):
    dataset = []
    labels = []
    for subject in subjects:
        for session in sessions:
            for trail in trails:
                target_file = f'{data_path}/{subject.value}_{session.value}_{trail}.mat'
                data = sio.loadmat(target_file)
                data_list = [data['chunks'][i] for i in range(data['chunks'].shape[0])]
                dataset.extend(data_list)
                labels.extend(data['labels'][0])

    return np.array(dataset), np.array(labels)


def pad_and_split(array, block_size, index):
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

    return blocks


def get_feature_dataset(data_path, subjects, sessions, trails, method, sample_length):
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

                    # data_transpose = np.transpose(trail_data, (2, 1, 0))
                    # chunks = pad_and_split(data_transpose, sample_length, 1)
                    data_transpose = np.transpose(trail_data, (1, 0, 2))
                    chunks = pad_and_split(data_transpose, sample_length, 0)

                    dataset.extend(chunks)
                    labels.extend([session_label[session][trail].value] * len(chunks))

                    found = True

                if not found:
                    raise ModuleNotFoundError

    return np.array(dataset), np.array(labels)
