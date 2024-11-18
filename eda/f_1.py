import copy

import numpy as np

from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torchsummary import summary
from scipy.stats import f_oneway

from data.seed_iv import Subject, FeatureMethod, Session
from pre.conformer import get_feature_dataset, dataset_of_subject
from model.conformer import Conformer


def analyze(config):
    subjects = [
        Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN,
        Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN,
        Subject.FOURTEEN, Subject.FIFTEEN
    ]
    sessions = [
        Session.ONE, Session.TWO, Session.THREE
    ]
    trails = list(range(0, 24))
    method = FeatureMethod.DE_LDS
    block_size = 10

    dataset, labels = get_feature_dataset(
        config.dataset['eeg_feature_smooth_abs_path'],
        subjects,
        sessions,
        trails,
        method,
        block_size
    )

    dataset_mean = dataset.mean(axis=(1, 2))








import matplotlib.pyplot as plt

# Set font properties globally for consistent appearance
plt.rcParams.update({
    'font.size': 12,  # General font size
    'font.family': 'serif',  # Use a serif font (e.g., Times New Roman)
    'axes.titlesize': 14,  # Title font size
    'axes.labelsize': 12,  # Label font size
    'xtick.labelsize': 10,  # X-axis tick labels
    'ytick.labelsize': 10,  # Y-axis tick labels
})
# %%
import pre.conformer
import os

eeg_feature_data_path = os.path.join(
    config.dataset['root_path'],
    config.dataset['eeg_feature_smooth_path']
)

for subj in Subject:
    sub_list = [subj]

    data_sub1_feat, label_sub1 = pre.conformer.get_feature_dataset(eeg_feature_data_path,
                                                                   sub_list,
                                                                   sessions=[Session.ONE, Session.TWO, Session.THREE],
                                                                   trails=list(range(0, 24)),
                                                                   method=FeatureMethod.DE_LDS,
                                                                   sample_length=10)
    from scipy.stats import f_oneway

    # Step 1: Average data over time samples and channels
    # Shape after mean: (570, 5) - 570 chunks, 5 frequency bands
    data_mean = data_sub1_feat.mean(axis=(1, 2))
    # Step 2: Separate data into groups by emotion labels
    emotion_groups = [data_mean[label_sub1 == i] for i in range(4)]

    # Step 3: Perform ANOVA (F-test) for each frequency band
    results = {}
    for band in range(5):  # Loop over frequency bands
        # Extract data for this frequency band across all emotion groups
        band_data = [group[:, band] for group in emotion_groups]
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*band_data)
        results[f"Frequency Band {band + 1}"] = {"F-statistic": f_stat, "p-value": p_value}

    # Step 4: Display results
    for band, stats in results.items():
        print(f"{band}: F-statistic = {stats['F-statistic']:.4f}, p-value = {stats['p-value']:.4e}")

    import matplotlib.pyplot as plt

    # Step 5: Extract results for visualization
    bands = list(results.keys())
    f_stats = [stats["F-statistic"] for stats in results.values()]
    p_values = [stats["p-value"] for stats in results.values()]

    # Step 6: Create the bar plot for F-statistics
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bands, f_stats, color='skyblue', edgecolor='black')

    # Annotate the bars with p-values
    for bar, p_value in zip(bars, p_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"p={p_value:.2e}", ha='center', va='bottom', fontsize=10)

    # Add labels, title, and grid
    plt.xlabel("Frequency Bands", fontsize=12)
    plt.ylabel("F-statistic", fontsize=12)
    plt.title(f"ANOVA Results Across Frequency Bands of Subject {sub_list[0].value}", fontsize=14)
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')  # Reference line
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Step 1: Average over time samples and channels
    data_mean = data_sub1_feat.mean(axis=(1, 2))  # Shape: (570, 5)

    # Step 2: Group data by emotion labels
    emotion_groups = [data_mean[label_sub1 == i] for i in range(4)]

    # Step 3: Perform F-test for each frequency band
    results = {}
    for band in range(5):  # Loop over frequency bands
        # Extract data for this frequency band across all emotion groups
        band_data = [group[:, band] for group in emotion_groups]
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*band_data)
        results[f"Frequency Band {band + 1}"] = {"F-statistic": f_stat, "p-value": p_value}

    # Step 4: Visualize results for each band
    plt.figure(figsize=(14, 8))
    for band in range(5):
        # Extract data for this band
        band_data = [group[:, band] for group in emotion_groups]

        # Create subplot for each band
        plt.subplot(2, 3, band + 1)
        plt.boxplot(band_data, tick_labels=["Emotion 0", "Emotion 1", "Emotion 2", "Emotion 3"], patch_artist=True)

        # Add F-statistic and p-value as the title
        f_stat = results[f"Frequency Band {band + 1}"]["F-statistic"]
        p_value = results[f"Frequency Band {band + 1}"]["p-value"]
        plt.title(f"Freq Band {band + 1}\nF-stat: {f_stat:.2f}, p: {p_value:.2e}")

        # Add labels
        plt.xlabel("Emotion Labels")
        plt.ylabel("Mean Value")

    # Adjust layout and show plot
    plt.tight_layout()
    plt.suptitle(f"F-Test Results Across Frequency Bands of Subject {sub_list[0].value}", fontsize=16, y=1.02)
    plt.show()