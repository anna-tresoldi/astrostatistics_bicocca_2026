# Galaxy vs QSO classifier (colors)

Quick instructions to run the classifier script that trains a small MLP on color features (u-g, g-r, r-i, i-z) from `solutions/galaxyquasar.csv`.

Install dependencies (recommended in a virtualenv):

```bash
pip install -r requirements.txt
```

Train with defaults:

```bash
python working/galaxy_qso_classifier.py
```

The trained model and scaler are saved to `working/galaxy_qso_model.pth` by default.
