import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_data(path):
    df = pd.read_csv(path)
    # construct color features u-g, g-r, r-i, i-z
    df = df.dropna() # drop rows with missing values
    feats = np.vstack([ # compute colors
        df['u'] - df['g'],
        df['g'] - df['r'],
        df['r'] - df['i'],
        df['i'] - df['z'],
    ]).T
    # labels: GALAXY -> 0, QSO -> 1
    labels = (df['class'] == 'QSO').astype(int).values
    return feats, labels


def train(args):
    X, y = load_data(args.csv) # load data and construct features/labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # stratify to maintain class balance in train/test split
    )

    scaler = StandardScaler() # scale data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TabularDataset(X_train, y_train)
    test_ds = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(n_features=X.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        preds = []
        ys = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                p = logits.argmax(dim=1).cpu().numpy()
                preds.append(p)
                ys.append(yb.numpy())

        preds = np.concatenate(preds)
        ys = np.concatenate(ys)
        acc = accuracy_score(ys, preds)
        print(f"Epoch {epoch}/{args.epochs} — train_loss={np.mean(losses):.4f} test_acc={acc:.4f}")

    # final evaluation and report
    print('\nFinal evaluation:')
    print(classification_report(ys, preds, target_names=['GALAXY', 'QSO']))

    # save artifacts
    torch.save({'model_state_dict': model.state_dict(), 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_}, args.output)
    print(f"Saved model + scaler to {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description='Galaxy vs QSO classifier (colors)')
    p.add_argument('--csv', type=str, default='solutions/galaxyquasar.csv', help='path to CSV file')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--output', type=str, default='working/galaxy_qso_model.pth')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
