import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- USER CONFIG ---
file_path = r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\CorrectedSim.xlsx"
sheet_name = "Meas"    # sheet name
row_actual = 0         # first row index (0-based)
row_pred = 1           # second row index (0-based)

# --- READ EXCEL ---
df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
y_true = df.iloc[row_actual].values
y_pred = df.iloc[row_pred].values

# Convert to numeric if possible
try:
    y_true = pd.to_numeric(y_true)
    y_pred = pd.to_numeric(y_pred)
    is_regression = True
except:
    is_regression = False

# --- REGRESSION METRICS ---
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mbe = np.mean(y_pred - y_true)  # <-- Bias Mean Error
    r2 = r2_score(y_true, y_pred)
    
    print("Regression Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Bias Mean Error (MBE): {mbe:.4f}")
    print(f"R²: {r2:.4f}")

# --- CLASSIFICATION METRICS ---
def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("Classification Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# --- AUTOMATICALLY DETECT AND RUN ---
try:
    if is_regression and any([isinstance(x, (float, int)) for x in y_true]):
        regression_metrics(y_true, y_pred)
    else:
        classification_metrics(y_true, y_pred)
except Exception as e:
    print("Error computing metrics:", e)
