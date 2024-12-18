"""
Training and testing models based on different experiment requirements.

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

import copy

import numpy as np

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torchsummary import summary

from data.seed_iv import Subject, FeatureMethod, Session, Band
from pre.conformer import get_feature_dataset
from model.conformer import Conformer


def a_model(config):
    """
    experiment A

    With shuffle_test: False
    Out of the 24 trials used in sample collection, the first 16 trials are the training data,
    and the last 8 trials are the test data.

    With shuffle_test: True
    Shuffle all the data, select two-thirds as the training set, and one-third as the test set.
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
    block_size = 10
    input_channels = len(bands)
    dim = 40
    heads = 10
    depth = 2
    method = FeatureMethod.DE_LDS
    all_trails = list(range(0, 24))
    train_trials = list(range(0, 16))
    test_trails = list(range(16, 24))
    batch_size = 1
    shuffle_test = True
    shuffle_spilt = 3
    num_epochs = 150
    best_accuracy = 0.8

    model = Conformer(
        input_channels=input_channels,
        block_size=block_size,
        dim=dim,
        heads=heads,
        depth=depth,
        classes=4
    )

    # show model summary
    if config.conformer['summary']:
        m = copy.copy(model)
        summary(m, input_size=(input_channels, block_size, 62))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    if shuffle_test:
        all_dataset, all_labels = get_feature_dataset(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            sessions,
            all_trails,
            method,
            block_size,
            bands
        )
        kf = KFold(n_splits=shuffle_spilt, shuffle=True, random_state=42)
        train_index, test_index = next(kf.split(all_dataset))
        x_train, x_test = np.array(all_dataset[train_index]), np.array(all_dataset[test_index])
        y_train, y_test = np.array(all_labels[train_index]), np.array((all_labels[test_index]))
    else:
        x_train, y_train = get_feature_dataset(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            sessions,
            train_trials,
            method,
            block_size,
            bands
        )
        x_test, y_test = get_feature_dataset(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            sessions,
            test_trails,
            method,
            block_size,
            bands
        )

    return model, x_train, x_test, y_train, y_test, criterion, optimizer, num_epochs, batch_size, best_accuracy


def b_model(config):
    """
    experiment B

    With shuffle_test: False
    Out of the 3 sessions used in sample collection, the data of one session are used as the training set,
    and the data of another session are used as the test set.

    With shuffle_test: True
    Shuffle two sessions' data, select half as the training set, and another half as the test set.
    """
    subjects = [
        Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN,
        Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN,
        Subject.FOURTEEN, Subject.FIFTEEN
    ]
    sessions = [
        Session.ONE, Session.TWO
    ]
    train_sessions = [
        Session.ONE
    ]
    test_sessions = [
        Session.TWO
    ]
    bands = [Band.DELTA, Band.THETA, Band.ALPHA, Band.BETA, Band.GAMMA]
    block_size = 10
    input_channels = len(bands)
    dim = 40
    heads = 10
    depth = 2
    method = FeatureMethod.DE_LDS
    all_trails = list(range(0, 24))
    batch_size = 1
    shuffle_test = True
    shuffle_spilt = 2
    num_epochs = 150
    best_accuracy = 0.7

    model = Conformer(
        input_channels=input_channels,
        block_size=block_size,
        dim=dim,
        heads=heads,
        depth=depth,
        classes=4
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    if shuffle_test:
        all_dataset, all_labels = get_feature_dataset(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            sessions,
            all_trails,
            method,
            block_size,
            bands,
        )
        kf = KFold(n_splits=shuffle_spilt, shuffle=True, random_state=42)
        train_index, test_index = next(kf.split(all_dataset))
        x_train, x_test = np.array(all_dataset[train_index]), np.array(all_dataset[test_index])
        y_train, y_test = np.array(all_labels[train_index]), np.array((all_labels[test_index]))
    else:
        x_train, y_train = get_feature_dataset(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            train_sessions,
            all_trails,
            method,
            block_size,
            bands
        )
        x_test, y_test = get_feature_dataset(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            test_sessions,
            all_trails,
            method,
            block_size,
            bands
        )

    return model, x_train, x_test, y_train, y_test, criterion, optimizer, num_epochs, batch_size, best_accuracy


def start(config):
    if config.conformer['experiment'] == 'A':
        model, x_train, x_test, y_train, y_test, criterion, optimizer, epochs, batch_size, best_accuracy = a_model(
            config
        )
    elif config.conformer['experiment'] == 'B':
        model, x_train, x_test, y_train, y_test, criterion, optimizer, epochs, batch_size, best_accuracy = b_model(
            config
        )
    else:
        raise NotImplementedError

    # GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x_train_tensor = torch.from_numpy(np.array(x_train)).float()
    y_train_tensor = torch.from_numpy(np.array(y_train)).long()
    x_test_tensor = torch.from_numpy(np.array(x_test)).float()
    y_test_tensor = torch.from_numpy(np.array(y_test)).long()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()

        print(f'Epoch: {epoch} {"-" * 55}')

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
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

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
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                model.state_dict(),
                f'./saved/{config.conformer["experiment"]}_{test_accuracy:.4f}.pth'
            )
            print(f'Epoch {epoch + 1}: {config.conformer["experiment"]} model saved with accuracy {test_accuracy:.4f}')

    print("-" * 64)

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
