# Credit Card Approval Prediction System

## Step 1: Extract the Dataset

⚠️ **Important**

Before running the project, extract the dataset ZIP file.

After extracting, make sure the `dataset` folder contains the following files:

```
dataset/
├── application_record.csv
└── credit_record.csv
```

If these files are missing, the project will not run.

---

## Step 2: Create a Virtual Environment

```bash
py -m venv venv
```

---

## Step 3: Activate the Virtual Environment

```bash
venv\Scripts\activate
```

---

## Step 4: Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Step 5: Train the Machine Learning Model

```bash
python train_model.py
```

Wait until the message below appears:

```
TRAINING COMPLETED SUCCESSFULLY
```

This will generate:

- model.pkl
- model_metadata.pkl
- model_comparison.csv

---

## Step 6: Run the Flask Application

```bash
python app.py
```

---

## Step 7: Open the Application

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Before Running, Verify the Following

✅ Dataset ZIP file has been extracted.

✅ `dataset` folder contains:
- `application_record.csv`
- `credit_record.csv`

✅ Required packages are installed.

✅ `train_model.py` has been executed successfully.

✅ `model.pkl` and `model_metadata.pkl` are created before running `app.py`.
