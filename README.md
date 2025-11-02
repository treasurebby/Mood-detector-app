# Mood-detector-model — quick setup

This project trains an emotion-detection CNN and provides a Flask web app to run predictions.

Quick automated setup (PowerShell)

1. Open PowerShell and cd into the project root:

```powershell
cd 'C:\Users\DELL LATITUDE 5320\Mood-detector-model'
```

2. Run the helper script to create a virtual environment, install dependencies and start a short training run:

```powershell
# test run: 1 epoch, small batch
.\setup_and_train.ps1 -Epochs 1 -BatchSize 16

# or full run (example)
.\setup_and_train.ps1 -Epochs 25 -BatchSize 64
```

Notes & troubleshooting
- If you already have a virtual env and want to skip installing packages, run:

```powershell
.\setup_and_train.ps1 -NoInstall -Epochs 1 -BatchSize 16
```

- If package installations fail on Windows with binary/DLL errors, try installing the Microsoft Visual C++ Redistributable (2015-2022 x64) and re-run the script.
- Use the project `requirements.txt` to control exact dependency versions.

If you prefer manual steps, see below:

```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip/build tooling
python -m pip install --upgrade pip setuptools wheel

# Install pinned dependencies
python -m pip install --no-cache-dir -r requirements.txt

# Run a short verification training
python model.py --epochs 1 --batch-size 16
```

If you run into import errors (e.g. ModuleNotFoundError), install the missing package with `python -m pip install <package>` and retry.

When training completes successfully a model file `emotion_detector_model.h5` will be created in the project root.
