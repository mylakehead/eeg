import numpy as np

from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torchsummary import summary

from data.seed_iv import Subject, session_label
from pre.data import subject_file_map, process_file
from model.vit import ViT


def dataset_of_subject(folder, subject):
    subject_file_mapping = subject_file_map(folder)
    files = subject_file_mapping[subject]

    total_data_set = list()
    total_labels = list()
    for k, v in files.items():
        file = files[k]
        labels = session_label[k]
        data_set, data_labels = process_file(file, labels, fs=200, chunk_length=700)

        total_data_set.extend(data_set)
        total_labels.extend(data_labels)

    return total_data_set, total_labels


def start(config):
    m = ViT()
    summary(m, input_size=(1, 62, 700))

    x, y = dataset_of_subject(config.dataset['eeg_raw_data_abs_path'], Subject.THREE)
    x = np.array(x)
    y = np.array(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        print(f"Fold {fold + 1}")

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train_tensor = torch.from_numpy(np.array(x_train)).float().unsqueeze(1)
        y_train_tensor = torch.from_numpy(np.array(y_train)).long()
        x_test_tensor = torch.from_numpy(np.array(x_test)).float().unsqueeze(1)
        y_test_tensor = torch.from_numpy(np.array(y_test)).long()

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = ViT()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

        model.train()
        for epoch in range(5):
            print(f'Epoch: {epoch} ----------------------------')

            running_loss = 0.0

            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()

                outputs = model(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    # print(f'Epoch [{epoch + 1}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                y_pred = model(batch_x)
                _, predicted = torch.max(y_pred, 1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

        print("-" * 40)

    print("k-fold ned")
