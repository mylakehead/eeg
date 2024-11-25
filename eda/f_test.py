"""
Module: EEG Feature Analysis with F-Test and Visualization

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

import matplotlib.pyplot as plt

from data.seed_iv import Subject, FeatureMethod, Session, Emotion, Band
from pre.conformer import get_feature_dataset


def analyze(config):
    """
    Perform exploratory data analysis (EDA) by analyzing EEG feature data
    and performing F-tests for various frequency bands.

    Args:
        config (Config): Configuration object containing dataset paths and parameters.
    """
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

    # Group data by emotion
    groups = {}
    for index, label in enumerate(labels):
        chunk = dataset[index, :, :, :]
        if Emotion(label) in groups:
            groups[Emotion(label)] = np.concatenate((groups[Emotion(label)], chunk), axis=1)
        else:
            groups[Emotion(label)] = chunk

    # Perform analysis for each frequency band
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

        # Compute the overall mean
        overall_mean = total/num

        # Compute F-test statistics
        k = len(band_group)
        s_b2 = sum(len(group) * (np.mean(group) - overall_mean) ** 2 for group in band_group) / (k - 1)
        s_w2 = sum(sum((x - np.mean(group)) ** 2 for x in group) for group in band_group) / (num - k)

        f_statistic = s_b2 / s_w2

        # Plot the results
        plt.figure(figsize=(11, 6))
        plt.boxplot(band_group, labels=[f'{Emotion(i)}'[8:] for i in range(k)])
        plt.title(f'{band}'[5:] + f'  F-test statistic: {f_statistic:.2f}')
        plt.axhline(overall_mean, color='red', linestyle='--', label='Overall Mean')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
