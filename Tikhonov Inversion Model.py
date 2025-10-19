import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# === 1. Load dataset ===
file_path = r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\Tiknohoc5.xlsx"
df = pd.read_excel(file_path)

# Separate features and target
X = df.iloc[:, :-1].values  # 151 frequency columns
y_true = df.iloc[:, -1].values  # target conductivity

# === 2. Normalize features to zero mean, unit variance ===
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# === 3. Initialize parameters ===
n_features = X_norm.shape[1]
m = np.zeros(n_features)         # initial guess for conductivity coefficients
lambda_reg = 0.1                 # Tikhonov damping (regularization strength)
alpha = 1e-5                     # small step size to ensure stable updates
n_iter = 500                      # number of iterations

# === 4. Iterative Tikhonov inversion (gradient descent) ===
for i in range(n_iter):
    # Predicted conductivity from current coefficients
    y_pred = X_norm @ m
    
    # Residual
    residual = y_true - y_pred
    
    # Gradient with Tikhonov regularization
    grad = -2 * (X_norm.T @ residual) + 2 * (lambda_reg ** 2) * m
    
    # Optional: clip extreme gradients for stability
    grad = np.clip(grad, -1e3, 1e3)
    
    # Update coefficients
    m -= alpha * grad
    
    # Print progress every 20 iterations
    if (i+1) % 20 == 0:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"Iteration {i+1}/{n_iter}: MAE={mae:.5f}, RMSE={rmse:.5f}")

# === 5. Final predictions ===
y_pred = X_norm @ m
y_pred = np.maximum(y_pred, 0)

# Evaluate performance
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\nFinal MAE: {mae:.5f}")
print(f"Final RMSE: {rmse:.5f}")

# === 6. Save results to Excel ===
df["Predicted_Conductivity"] = y_pred
output_file = "tikhonov_predictions_stable5.xlsx"
df.to_excel(output_file, index=False)
print(f"Predictions saved to: {output_file}")

# === 7. Optional visualization ===
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("True Conductivity (S/m)")
plt.ylabel("Predicted Conductivity (S/m)")
plt.title("Stable Iterative Tikhonov Linear Inversion")
plt.show()
