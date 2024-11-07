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
# from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchsummary import summary

from data.seed_iv import session_label, Subject, FeatureMethod
from pre.seed_iv import subject_file_map, process_feature_file, process_feature_file_for_experiment_a
from model.conformer import Conformer


def dataset_of_experiment_a(folder, subjects, feature_method: FeatureMethod, block_size: int):
    total_train_data_set = list()
    total_train_labels = list()
    total_test_data_set = list()
    total_test_labels = list()

    for subject in subjects:
        subject_file_mapping = subject_file_map(folder)
        files = subject_file_mapping[subject]

        for k, v in files.items():
            file = files[k]
            labels = session_label[k]
            train_data_set, train_data_labels, test_data_set, test_data_labels = process_feature_file_for_experiment_a(
                feature_method, file, labels, block_size
            )

            total_train_data_set.extend(train_data_set)
            total_train_labels.extend(train_data_labels)
            total_test_data_set.extend(test_data_set)
            total_test_labels.extend(test_data_labels)

    return (
        np.array(total_train_data_set), np.array(total_train_labels),
        np.array(total_test_data_set), np.array(total_test_labels)
    )


def dataset_of_subject(folder, subjects, feature_method, block_size) -> tuple[list[np.ndarray], list[int]]:
    subject_file_mapping = subject_file_map(folder)

    total_data_set = list()
    total_labels = list()

    for subject in subjects:
        files = subject_file_mapping[subject]
        for k, v in files.items():
            file = files[k]
            labels = session_label[k]
            data_set, data_labels = process_feature_file(feature_method, file, labels, block_size)

            total_data_set.extend(data_set)
            total_labels.extend(data_labels)

    return total_data_set, total_labels


def start(config):
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")
        
    if config.conformer['experiment'] == "A":
        block_size = 5
        method = FeatureMethod.DE_LDS
        x_train, y_train, x_test, y_test = dataset_of_experiment_a(
            config.dataset['eeg_feature_smooth_abs_path'],
            [Subject.THREE],
            method,
            block_size,
        )
        model = Conformer(channels=5, block_size=block_size, dim=1, heads=1, depth=1, classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    elif config.conformer['experiment'] == "B":
        block_size = 10
        method = FeatureMethod.DE_LDS
        x_train, y_train, x_test, y_test = dataset_of_experiment_a(
            config.dataset['eeg_feature_smooth_abs_path'],
            [Subject.THREE],
            method,
            block_size,
        )
        model = Conformer(channels=5, block_size=block_size, dim=40, heads=10, depth=6, classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif config.conformer['experiment'] == "C":
        block_size = 10
        method = FeatureMethod.DE_LDS
        x, y = dataset_of_subject(
            config.dataset['eeg_feature_smooth_abs_path'],
            [Subject.THREE],
            method, block_size
        )
        x = np.array(x)
        y = np.array(y)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_index, test_index = next(kf.split(x))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Conformer(channels=5, block_size=block_size, dim=40, heads=10, depth=6, classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        block_size = 10
        method = FeatureMethod.DE_LDS
        x, y = dataset_of_subject(
            config.dataset['eeg_feature_smooth_abs_path'],
            [
                Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE,
                Subject.SIX, Subject.SEVEN, Subject.EIGHT, Subject.NINE, Subject.TEN,
                Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN, Subject.FOURTEEN, Subject.FIFTEEN
            ],
            method, block_size
        )
        x = np.array(x)
        y = np.array(y)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_index, test_index = next(kf.split(x))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Conformer(channels=5, block_size=block_size, dim=30, heads=5, depth=4, classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    if config.conformer['summary']:
        m = Conformer(5, block_size=block_size)
        summary(m, input_size=(10, 62, 5))

    x_train_tensor = torch.from_numpy(np.array(x_train)).float()
    y_train_tensor = torch.from_numpy(np.array(y_train)).long()
    x_test_tensor = torch.from_numpy(np.array(x_test)).float()
    y_test_tensor = torch.from_numpy(np.array(y_test)).long()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.8
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()

        print(f'Epoch: {epoch} ----------------------------')
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

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
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                y_pred = model(batch_x)
                loss = criterion(y_pred, batch_y)
                test_loss += loss.item()
                _, predicted = torch.max(y_pred, 1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = correct / total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # scheduler.step(test_loss)
        # current_lr = scheduler.get_last_lr()[0]
        # print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if config.conformer['experiment'] == "PRE":
                torch.save(model.state_dict(), f'./saved/pre_{test_accuracy:.4f}.pth')
            else:
                pass
            print(f"Epoch {epoch + 1}: New best model saved with accuracy {test_accuracy:.4f}")

    print("-" * 40)
    return
