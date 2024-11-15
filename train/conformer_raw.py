import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from data.seed_iv import Subject, Session
from model.conformer import Conformer
from pre.conformer import get_raw_dataset


def start(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = config.conformer["experiment"]
    raw_processed_path = config.dataset['raw_processed_path']

    if experiment == "A":
        subjects = [
            Subject.THREE
        ]
        sessions = [
            Session.ONE, Session.TWO, Session.THREE
        ]
        train_trials = list(range(0, 16))
        test_trails = list(range(16, 24))
        batch_size = 200

        best_accuracy = 0.8

        model = Conformer(emb_size=40, inner_channels=40, heads=10, depth=6, n_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    else:
        raise NotImplementedError

    model.to(device)

    train_dataset, train_labels = get_raw_dataset(
        raw_processed_path,
        subjects,
        sessions,
        train_trials
    )
    test_dataset, test_labels = get_raw_dataset(
        raw_processed_path,
        subjects,
        sessions,
        test_trails
    )
    x_train_tensor = torch.from_numpy(train_dataset).float().unsqueeze(1)
    y_train_tensor = torch.from_numpy(train_labels).long()
    x_test_tensor = torch.from_numpy(test_dataset).float().unsqueeze(1)
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

        test_loss = test_loss / len(test_loader)
        test_accuracy = correct / total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if len(subjects) == 1:
                torch.save(
                    model.state_dict(),
                    f'./saved/{experiment}_train_accuracy_{epoch_accuracy:.4f}_test_accuracy_{test_accuracy:.4f}.pth'
                )
            else:
                torch.save(
                    model.state_dict(),
                    f'./saved/{experiment}__test_accuracy_{test_accuracy:.4f}.pth'
                )
            print(f"Epoch {epoch + 1}: New {experiment} model saved with accuracy {test_accuracy:.4f}")

    print("-" * 40)
    return

