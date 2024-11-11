import numpy as np

import scipy.io as sio

from data.seed_iv import session_label


def get_dataset(data_path, subjects, sessions, trails):
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
