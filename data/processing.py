import os
import re

import numpy as np
import scipy.io as sio
from scipy.signal import cheby2, filtfilt

from seed_iv import Subject, subject_file_map, session_label


source_dir = "../.dataset/SEED_IV"
target_dir = "../.dataset/SEED_IV_PROCESSED"

fc = 200
Wl = 4
Wh = 47
Wn = [Wl * 2 / fc, Wh * 2 / fc]
ba = cheby2(6, 60, Wn, btype='bandpass')
b = ba[0]
a = ba[1]

chunk_size = 200


subjects = [
    Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE,
    Subject.SIX, Subject.SEVEN, Subject.EIGHT, Subject.NINE, Subject.TEN,
    Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN, Subject.FOURTEEN, Subject.FIFTEEN
]


if __name__ == '__main__':
    for subject in subjects:
        subject_file_mapping = subject_file_map(source_dir)
        files = subject_file_mapping[subject]

        for k, v in files.items():
            file = files[k]
            labels = session_label[k]

            data = sio.loadmat(file)
            data.pop('__header__', None)
            data.pop('__version__', None)
            data.pop('__globals__', None)

            pattern = r'[a-zA-Z]+_eeg(\d+)'

            must = list(range(1, 25))
            for index, trial in data.items():
                match = re.match(pattern, index)
                if not match:
                    continue

                trail_index = int(match.group(1))
                label_index = trail_index - 1
                print(f'processing {subject} {k} trail: {trail_index}...')

                trial_mean = np.mean(trial, axis=1, keepdims=True)
                trial_std = np.std(trial, axis=1, ddof=1, keepdims=True)
                trial_data = (trial - trial_mean) / trial_std

                trial_data = filtfilt(b, a, trial_data, axis=1)
                trial_label = labels[label_index]

                chunks = []
                chunk_labels = []
                chunk_num = np.int32(trial_data.shape[1] / 200)
                for i in range(chunk_num):
                    chunks.append(trial_data[:, i * 200:(i + 1) * 200])
                chunk_labels = [trial_label.value] * chunk_num

                target_file = f'{target_dir}/{subject.value}_{k.value}_{label_index}.mat'
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                sio.savemat(target_file, {'chunks': chunks, 'labels': chunk_labels})

                must.remove(trail_index)

            if len(must) != 0:
                raise Exception(f"file missing{must}")

