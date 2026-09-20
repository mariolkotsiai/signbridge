import os
import sys

# Εξασφάλιση ότι το root directory μπαίνει πρώτο στο sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    from ml.models.sign_model import SignLanguageLSTM
except ModuleNotFoundError:
    # Fallback σε περίπτωση που εκτελείται από διαφορετικό Working Directory
    from models.sign_model import SignLanguageLSTM


class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(language='gsl', epochs=20, lr=0.001):
    dataset_dir = os.path.join(PROJECT_ROOT, f"ml/datasets/{language}/raw_landmarks")
    file_paths = glob.glob(os.path.join(dataset_dir, "*.npy"))

    if not file_paths:
        print(f"[ERROR] Δεν βρέθηκαν .npy αρχεία στον φάκελο: {dataset_dir}")
        return

    # Συλλογή labels
    labels_set = sorted(list(set([os.path.basename(p).split('_sample_')[0].split('_open_')[0] for p in file_paths])))
    label_to_id = {label: i for i, label in enumerate(labels_set)}

    X, y = [], []
    for p in file_paths:
        data = np.load(p)
        label_name = os.path.basename(p).split('_sample_')[0].split('_open_')[0]
        if label_name in label_to_id:
            X.append(data)
            y.append(label_to_id[label_name])

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(LandmarkDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(LandmarkDataset(X_val, y_val), batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageLSTM(num_classes=len(labels_set)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Έναρξη Εκπαίδευσης Μοντέλου [{language.upper()}] ({len(labels_set)} κλάσεις) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_acc = (correct / total) * 100 if total > 0 else 0.0
        print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.1f}%")

    # Αποθήκευση Μοντέλου
    weights_dir = os.path.join(PROJECT_ROOT, "services/inference/weights")
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, f"{language}_model.pth")
    torch.save({'model_state': model.state_dict(), 'labels': labels_set}, model_path)
    print(f"\n[SUCCESS] Το μοντέλο αποθηκεύτηκε στο: {model_path}")


if __name__ == "__main__":
    train_model(language='gsl', epochs=15)
    train_model(language='asl', epochs=15)