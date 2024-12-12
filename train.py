import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import constants as c
from serializers import CSVSerializer

LABEL_COLUMN = "label_bullish_0"
DROP_COLUMNS = [
    "timestamp",
    "close_time",
    "label_bullish_0",
    "label_bullish_1",
    "label_bullish_2",
    "label_bullish_3",
    "label_bullish_4",
    "label_bearish_0",
    "label_bearish_1",
    "label_bearish_2",
    "label_bearish_3",
    "label_bearish_4",
]
DROP_COLUMNS.remove(LABEL_COLUMN)


class CryptoPriceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AdvancedNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def sequential_split(df, test_size=0.2):
    total_length = len(df)
    test_samples = int(total_length * test_size)
    train_samples = total_length - test_samples

    x_train = df.iloc[:train_samples].drop([LABEL_COLUMN], axis=1)
    y_train = df.iloc[:train_samples][LABEL_COLUMN]

    x_test = df.iloc[train_samples:].drop([LABEL_COLUMN], axis=1)
    y_test = df.iloc[train_samples:][LABEL_COLUMN]

    return x_train, x_test, y_train, y_test


def feature_selection(X, y, top_features=20):
    mi_scores = mutual_info_classif(X, y)
    feature_importance = dict(zip(X.columns, mi_scores))

    selected_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_features]

    selected_feature_names = [f[0] for f in selected_features]
    return X[selected_feature_names], selected_feature_names


def train_model(features, labels):
    # features_selected, selected_features = feature_selection(features, labels)

    x_train, x_test, y_train, y_test = sequential_split(pd.concat([features, labels], axis=1), test_size=0.2)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_train = y_train.values if hasattr(y_train, "values") else y_train
    y_test = y_test.values if hasattr(y_test, "values") else y_test

    train_dataset = CryptoPriceDataset(x_train_scaled, y_train.astype(float))
    test_dataset = CryptoPriceDataset(x_test_scaled, y_test.astype(float))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedNeuralNetwork(x_train.shape[1]).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        total_loss = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)

                predictions = model(batch_features)
                predicted_labels = (predictions > 0.5).float()

                correct += (predicted_labels == batch_labels).sum().item()
                total += batch_labels.size(0)

            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/50], Accuracy: {accuracy:.2f}%")

    return model, scaler


def clean_dataframe(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())
    return df


def main():
    csv_serializer = CSVSerializer(output_dir=c.CSV_OUTPUT_DIR_PATH)
    for interval in c.TimeInterval:
        print(f"Training {interval.value} Data")
        df = csv_serializer.from_csv(f"btc_{interval.value}.csv")
        df.drop(DROP_COLUMNS, axis=1, inplace=True)
        df = clean_dataframe(df)
        features = df.drop([LABEL_COLUMN], axis=1)
        labels = df[LABEL_COLUMN]
        train_model(features, labels)


if __name__ == "__main__":
    main()
