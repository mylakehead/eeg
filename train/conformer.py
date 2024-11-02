"""
Module Name: EEG Data Classification with Conformer Model

Description:
    This module provides functionality for EEG data classification using a Conformer-based neural network model.
    The model is trained and evaluated on EEG data using k-fold cross-validation. The module includes functions
    for loading and preprocessing EEG data, and a main `start` function that initiates the training and evaluation
    process, tracking model performance across each fold.

Functions:
    - dataset_of_subject: Loads and processes EEG data for a specific subject and session, applying the specified
      feature extraction method and returning the data split into chunks with corresponding labels.
    - start: Main function to perform k-fold cross-validation on the Conformer model. It includes training and
      evaluation steps, printing out loss and accuracy metrics for each epoch and fold.

Usage:
    The module is designed for EEG-based machine learning tasks, specifically for classification tasks where
    data from multiple subjects is used. The Conformer model processes the EEG data and learns to classify it
    based on labeled chunks.

License:
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
    Created: 2024-10-16
    Last Modified: 2024-11-02
"""

import numpy as np

from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# from torchsummary import summary

from data.seed_iv import session_label, Subject, FeatureMethod
from pre.seed_iv import subject_file_map, process_feature_file
from model.conformer import Conformer


def dataset_of_subject(
        folder: str, subject: Subject, feature_method: FeatureMethod, block_size: int
) -> tuple[list[np.ndarray], list[int]]:
    """
    Loads and processes EEG data for a specific subject across multiple sessions, applying the specified
    feature extraction method and returning the data split into chunks with corresponding labels.

    :param folder: The root folder containing EEG data files organized by session.
    :type folder: str
    :param subject: The subject whose data is to be processed.
    :type subject: Subject
    :param feature_method: The feature extraction method to use, such as DE_LDS or PSD_LDS.
    :type feature_method: FeatureMethod
    :param block_size: The block size to split each trial into non-overlapping segments.
    :type block_size: int
    :return: A tuple containing two lists:
        - total_data_set: A list of np.ndarray blocks, each representing a chunk of EEG data.
        - total_labels: A list of integer labels corresponding to each chunk in `total_data_set`.
    :rtype: tuple[list[np.ndarray], list[int]]

    This function performs the following steps:
        1. Retrieves the file paths for the specified subject across all sessions from `subject_file_map`.
        2. Iterates over each session's file:
            - Loads the session labels from `session_label`.
            - Processes the feature file using `process_feature_file`, which applies the specified feature extraction
              method and splits the EEG data into chunks of the specified block size.
            - Appends the resulting data chunks and labels to `total_data_set` and `total_labels`, respectively.
        3. Returns `total_data_set` and `total_labels`, which collectively represent the processed data for the subject.

    Note:
        - This function is specific to a single subject, aggregating data across multiple sessions for that subject.
        - The function assumes that `subject_file_map` and `session_label` provide correct mappings for the
          files and labels, respectively.

    """
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


def start(config):
    """
    Main function to train and evaluate the Conformer model on EEG data using k-fold cross-validation.

    :param config: A configuration object containing dataset paths and other parameters for training.
    :type config: Config

    This function performs the following steps:
        1. Loads and processes the EEG dataset for a specified subject and feature extraction method.
        2. Converts the dataset into numpy arrays for easy indexing and manipulation.
        3. Performs k-fold cross-validation (default 5 splits):
            - Splits the data into training and testing sets for each fold.
            - Converts the training and testing data into PyTorch tensors and loads them into DataLoaders.
            - Initializes the Conformer model, loss criterion, and optimizer.
            - Trains the model over multiple epochs for each fold:
                - Tracks and prints the training loss and accuracy for each epoch.
            - Evaluates the model on the test set for each fold:
                - Calculates and prints the test loss and accuracy for each fold.
        4. Prints performance metrics for each fold to monitor the model's behavior and performance.

    Notes:
        - The model is trained using CrossEntropyLoss and optimized with the Adam optimizer.
        - The number of epochs and learning rate are hard-coded in this function but can be adjusted as needed.

    Parameters used within the function:
        - block_size (int): The block size for chunking EEG data. Default is 10.
        - num_epochs (int): The number of training epochs for each fold. Default is 1000.
        - learning_rate (float): The learning rate for the optimizer. Default is 0.0002.
    """
    block_size = 10
    x, y = dataset_of_subject(
        config.dataset['eeg_feature_smooth_abs_path'], Subject.THREE, FeatureMethod.DE_LDS, block_size
    )

    # m = ConformerFeature(5)
    # summary(m, input_size=(10, 62, 5))

    x = np.array(x)
    y = np.array(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        print(f"Fold {fold + 1}")

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train_tensor = torch.from_numpy(np.array(x_train)).float()
        y_train_tensor = torch.from_numpy(np.array(y_train)).long()
        x_test_tensor = torch.from_numpy(np.array(x_test)).float()
        y_test_tensor = torch.from_numpy(np.array(y_test)).long()

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = Conformer(channels=5, block_size=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

        num_epochs = 1000
        for epoch in range(num_epochs):
            model.train()

            print(f'Epoch: {epoch} ----------------------------')
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()

                outputs = model(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += predicted.eq(batch_y).sum().item()
                total_samples += batch_y.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_samples

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            model.eval()
            correct = 0
            total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    y_pred = model(batch_x)
                    loss = criterion(y_pred, batch_y)
                    test_loss += loss.item()
                    _, predicted = torch.max(y_pred, 1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()

            test_loss = test_loss / len(test_loader)
            test_accuracy = correct / total
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        print("-" * 40)
        return

    print("k-fold ned")
