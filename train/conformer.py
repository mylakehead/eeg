import copy

import numpy as np

from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torchsummary import summary

from data.seed_iv import Subject, FeatureMethod, Session
from pre.conformer import get_feature_dataset, dataset_of_subject
from model.conformer import Conformer


def a_model(config, old):
    subjects = [
        Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN,
        Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN,
        Subject.FOURTEEN, Subject.FIFTEEN
    ]
    sessions = [
        Session.ONE, Session.TWO, Session.THREE
    ]
    block_size = 10
    input_channels = 5
    dim = 40
    heads = 10
    depth = 6
    method = FeatureMethod.DE_LDS
    all_trails = list(range(0, 24))
    train_trials = list(range(0, 16))
    test_trails = list(range(16, 24))
    batch_size = 1
    shuffle_test = True
    shuffle_spilt = 3
    num_epochs = 1000

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
        )
        x_test, y_test = get_feature_dataset(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            sessions,
            test_trails,
            method,
            block_size,
        )

    if old:
        x, y = dataset_of_subject(
            config.dataset['eeg_feature_smooth_abs_path'], Subject.ONE, method, block_size
        )
        x = np.array(x)
        y = np.array(y)

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        train_index, test_index = next(kf.split(x))
        x_train, x_test = np.array(x[train_index]), np.array(x[test_index])
        y_train, y_test = np.array(y[train_index]), np.array(y[test_index])

    return model, x_train, x_test, y_train, y_test, criterion, optimizer, num_epochs, batch_size


def b_model(config):
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
    block_size = 10
    input_channels = 5
    dim = 40
    heads = 10
    depth = 6
    method = FeatureMethod.DE_LDS
    all_trails = list(range(0, 24))
    batch_size = 1
    shuffle_test = True
    shuffle_spilt = 2
    num_epochs = 1000

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
        )
        x_test, y_test = get_feature_dataset(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            test_sessions,
            all_trails,
            method,
            block_size,
        )

    return model, x_train, x_test, y_train, y_test, criterion, optimizer, num_epochs, batch_size


def start(config):
    if config.conformer['experiment'] == 'A':
        model, x_train, x_test, y_train, y_test, criterion, optimizer, epochs, batch_size = a_model(config, False)
    elif config.conformer['experiment'] == 'B':
        model, x_train, x_test, y_train, y_test, criterion, optimizer, epochs, batch_size = b_model(config)
    else:
        raise NotImplementedError

    if config.conformer['summary']:
        m = copy.copy(model)
        summary(m, input_size=(5, 10, 62))

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
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    print("-" * 64)
