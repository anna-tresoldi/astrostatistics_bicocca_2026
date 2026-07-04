"""
Professor-style PyTorch training script for galaxy vs QSO classification
Uses simple fully-connected network, SGD optimizer, ReduceLROnPlateau scheduler
and early stopping on validation loss. Saves trained model to a pickle file.

Run:
    python working/galaxy_qso_profstyle.py

"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata


class Net(nn.Module):
    def __init__(self, nhidden):
        super(Net, self).__init__()
        # input: 4 color features
        self.fc_h = nn.Linear(4, nhidden)
        self.fc_o = nn.Linear(nhidden, 1)  # single logit for binary classification

    def forward(self, x):
        h = F.relu(self.fc_h(x))
        logit = self.fc_o(h)
        return logit


def prepare_data(path):
    df = pd.read_csv(path).dropna()
    # features: colors u-g, g-r, r-i, i-z
    X = np.vstack([
        df['u'] - df['g'],
        df['g'] - df['r'],
        df['r'] - df['i'],
        df['i'] - df['z'],
    ]).T.astype(np.float32)
    y = (df['class'] == 'QSO').astype(np.float32).values.reshape(-1, 1)
    return X, y


def train_prof(path='solutions/galaxyquasar.csv', nhidden=8, epochs=100, batch_size=128):
    X, y = prepare_data(path)

    # normalize features to zero mean unit variance using full dataset (prof style)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    Xn = (X - mu) / sigma

    # 9:1 split (train:test)
    N = Xn.shape[0]
    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)
    ntrain = (N // 10) * 9
    train_idx, test_idx = idx[:ntrain], idx[ntrain:]

    X_train, y_train = Xn[train_idx], y[train_idx]
    X_test, y_test = Xn[test_idx], y[test_idx]

    train_ds = torchdata.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = torchdata.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = torchdata.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(nhidden).to(device)

    criterion = torch.nn.MSELoss(reduction='sum') #nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=5)#, verbose=True)

    best_val_loss = float('inf')
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader: 
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad() 
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        # validation loss on full test set
        model.eval()
        with torch.no_grad():
            Xte = torch.from_numpy(X_test).to(device)
            yte = torch.from_numpy(y_test).to(device)
            logits = model(Xte)
            val_loss = criterion(logits, yte).item()

        train_loss /= len(train_ds)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4e} val_loss={val_loss:.4e}")

        # early stopping logic
        if val_loss < best_val_loss * (1 - 1e-3):
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= 10:
            print("Early stopping: no improvement in validation loss")
            break

        scheduler.step(val_loss)

    # final evaluation
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        ytrue = y_test.flatten().astype(int)
        acc = (preds == ytrue).mean()
        from sklearn.metrics import classification_report
        print('\nFinal test accuracy: %.4f' % acc)
        print(classification_report(ytrue, preds, target_names=['GALAXY','QSO']))

    # save model + normalization
    out = {'model_state_dict': model.state_dict(), 'mu': mu, 'sigma': sigma}
    with open('working/galaxy_qso_prof_model.pkl', 'wb') as f:
        pickle.dump(out, f)
    print('Saved trained model to working/galaxy_qso_prof_model.pkl')

    return model


if __name__ == '__main__':
    train_prof()
