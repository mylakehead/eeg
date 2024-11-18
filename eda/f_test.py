import numpy as np

import matplotlib.pyplot as plt

from data.seed_iv import Subject, FeatureMethod, Session, Emotion, Band
from pre.conformer import get_feature_dataset


def analyze(config):
    subjects = [
        Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN,
        Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN,
        Subject.FOURTEEN, Subject.FIFTEEN
    ]
    sessions = [
        Session.ONE, Session.TWO, Session.THREE
    ]
    bands = [Band.DELTA, Band.THETA, Band.ALPHA, Band.BETA, Band.GAMMA]
    trails = list(range(0, 24))
    method = FeatureMethod.DE_LDS
    block_size = 10

    dataset, labels = get_feature_dataset(
        config.dataset['eeg_feature_smooth_abs_path'],
        subjects,
        sessions,
        trails,
        method,
        block_size,
        bands
    )

    groups = {}
    for index, label in enumerate(labels):
        chunk = dataset[index, :, :, :]
        if Emotion(label) in groups:
            groups[Emotion(label)] = np.concatenate((groups[Emotion(label)], chunk), axis=1)
        else:
            groups[Emotion(label)] = chunk

    for _, band in enumerate([Band.DELTA, Band.THETA, Band.ALPHA, Band.BETA, Band.GAMMA]):
        band_index = band.value

        band_group = []
        num = 0
        total = 0
        for _, emotion in enumerate([Emotion.NEUTRAL, Emotion.SAD, Emotion.FEAR, Emotion.HAPPY]):
            g = groups[emotion][band_index]
            g = np.mean(g, axis=1)
            band_group.append(g)
            num += len(g)
            total += np.sum(g)

        overall_mean = total/num

        k = len(band_group)
        s_b2 = sum(len(group) * (np.mean(group) - overall_mean) ** 2 for group in band_group) / (k - 1)
        s_w2 = sum(sum((x - np.mean(group)) ** 2 for x in group) for group in band_group) / (num - k)

        f_statistic = s_b2 / s_w2

        plt.figure(figsize=(11, 6))
        plt.boxplot(band_group, labels=[f'{Emotion(i)}'[8:] for i in range(k)])
        plt.title(f'{band}'[5:] + f'  F-test statistic: {f_statistic:.2f}')
        plt.axhline(overall_mean, color='red', linestyle='--', label='Overall Mean')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
