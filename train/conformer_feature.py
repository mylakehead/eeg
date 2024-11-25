import numpy as np

from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torchsummary import summary

from data.seed_iv import session_label, Subject, FeatureMethod, subject_file_map, Session
from pre.seed_iv import process_feature_file
from pre.conformer import get_feature_dataset
from model.conformer_feature import ConformerFeature
from model.sgcn import SignedGCNConv
from model.conformer_b import ConformerB


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = config.conformer["experiment"]
        
    if experiment == "PRE-A":
        block_size = 10
        dim = 40
        heads = 5
        depth = 4
        method = FeatureMethod.DE_LDS
        best_accuracy = 0.8

        subjects = [Subject.THREE]
        x_train, y_train, x_test, y_test = dataset_of_experiment_a(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            method,
            block_size,
        )
        model = ConformerFeature(channels=5, block_size=block_size, dim=dim, heads=heads, depth=depth, classes=4)
        model.load_state_dict(torch.load(config.conformer["A"]["PRE"]))
        for name, param in model.named_parameters():
            if "conv" in name or "transformer" in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    elif experiment == "A":
        subjects = [
            Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN,
            Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN,
            Subject.FOURTEEN, Subject.FIFTEEN
        ]
        sessions = [
            Session.ONE, Session.TWO, Session.THREE
        ]
        sample_length = 10
        train_trials = list(range(0, 16))
        test_trails = list(range(16, 24))
        method = FeatureMethod.DE_LDS
        batch_size = 200
        
        inner_channels = 10
        heads = 10
        depth = 3
        
        best_accuracy = 0.8

        model = ConformerFeature(
            input_channels=10,
            sample_length=10,
            inner_channels=inner_channels,
            heads=heads, depth=depth, classes=4
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif experiment == "B":
        subjects = [
            Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN,
            Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN,
            Subject.FOURTEEN, Subject.FIFTEEN
        ]
        sessions = [
            Session.ONE, Session.TWO, Session.THREE
        ]
        sample_length = 10
        train_trials = list(range(0, 16))
        test_trails = list(range(16, 24))
        method = FeatureMethod.DE_LDS
        batch_size = 200

        model = SignedGCNConv(in_channels=5, hidden_channels=16, out_channels=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif experiment == "C":
        block_size = 10
        dim = 40
        heads = 5
        depth = 6
        method = FeatureMethod.DE_LDS
        best_accuracy = 0.95

        subjects = [
            Subject.TWO
        ]

        x, y = dataset_of_subject(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            method, block_size
        )
        x = np.array(x)
        y = np.array(y)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_index, test_index = next(kf.split(x))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = ConformerFeature(channels=5, block_size=block_size, dim=dim, heads=heads, depth=depth, classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    elif experiment == 'D':
        block_size = 10
        dim = 30
        heads = 5
        depth = 4
        method = FeatureMethod.DE_LDS
        best_accuracy = 0.9
        batch_size = 200

        subjects = [
            Subject.THREE
        ]
        x, y = dataset_of_subject(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            method, block_size
        )
        x = np.array(x)
        y = np.array(y)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_index, test_index = next(kf.split(x))
        train_dataset, test_dataset = x[train_index], x[test_index]
        train_labels, test_labels = y[train_index], y[test_index]

        model = ConformerB(channels=5, block_size=block_size, dim=dim, heads=heads, depth=depth, classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    elif experiment == 'PRE':
        block_size = 10
        # model = Conformer(channels=5, block_size=block_size, dim=40, heads=5, depth=4, classes=4) 0.8+
        # model = Conformer(channels=5, block_size=block_size, dim=80, heads=20, depth=4, classes=4) 0.8+
        dim = 40
        heads = 10
        depth = 6
        method = FeatureMethod.DE_LDS
        best_accuracy = 0.85

        subjects = [
            Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN, 
            Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN, 
            Subject.FOURTEEN, Subject.FIFTEEN
        ]
        x, y = dataset_of_subject(
            config.dataset['eeg_feature_smooth_abs_path'],
            subjects,
            method, block_size
        )
        x = np.array(x)
        y = np.array(y)

        x_train = x
        y_train = y
        x_test = np.array([])
        y_test = np.array([])

        model = ConformerFeature(channels=5, block_size=block_size, dim=dim, heads=heads, depth=depth, classes=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    else:
        raise NotImplementedError

    model.to(device)

    if config.conformer['summary']:
        m = ConformerFeature(5, block_size=block_size)
        summary(m, input_size=(10, 62, 5))

    '''
    train_dataset, train_labels = get_feature_dataset(
        config.dataset['eeg_feature_smooth_abs_path'],
        subjects,
        sessions,
        train_trials,
        method,
        sample_length
    )
    test_dataset, test_labels = get_feature_dataset(
        config.dataset['eeg_feature_smooth_abs_path'],
        subjects,
        sessions,
        test_trails,
        method,
        sample_length
    )
    '''

    x_train_tensor = torch.from_numpy(train_dataset).float()
    y_train_tensor = torch.from_numpy(train_labels).long()
    x_test_tensor = torch.from_numpy(test_dataset).float()
    y_test_tensor = torch.from_numpy(test_labels).long()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

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

        if experiment != "PRE":
            test_loss = test_loss / len(test_loader)
            test_accuracy = correct / total
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                if len(subjects) == 1:
                    torch.save(
                        model.state_dict(),
                        f'./saved/{experiment}_method_{method.value}_train_accuracy_{epoch_accuracy:.4f}_test_accuracy_{test_accuracy:.4f}_{subjects[0]}_{sample_length}_{inner_channels}_{heads}_{depth}.pth'
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        f'./saved/{experiment}_method_{method.value}_test_accuracy_{test_accuracy:.4f}_{inner_channels}_{heads}_{depth}.pth'
                    )
                print(f"Epoch {epoch + 1}: New {experiment} model saved with accuracy {test_accuracy:.4f}")
        else:
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(
                    model.state_dict(),
                    f'./saved/{experiment}_{method.value}_{epoch_accuracy:.4f}_{dim}_{heads}_{depth}.pth'
                )
                print(f"Epoch {epoch + 1}: New {experiment} model saved with accuracy {epoch_accuracy:.4f}")

        # scheduler.step(test_loss)
        # current_lr = scheduler.get_last_lr()[0]
        # print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")

    print("-" * 40)
    return
