import numpy as np
import pandas as pd
import re
import json
import statsmodels.api as sm

# ---------- USER SETTINGS ----------
FILENAME = r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\Sim-Lab-Conversion.xlsx"
LAB_SHEET = "Lab"
SIM_SHEET = "Sim"
OUT_SHEET = "Sim_Corrected"
OUT_PARAMS = "phase_correction_model.json"
T0 = 37.0  # reference temperature
NOISE_SCALE = 0.5  # scale of lab residual noise to add
# ----------------------------------

def extract_freq(colname):
    s = str(colname)
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None

# === LOAD DATA ===
lab_df = pd.read_excel(FILENAME, sheet_name=LAB_SHEET)
sim_df = pd.read_excel(FILENAME, sheet_name=SIM_SHEET)

# Identify frequency columns
freq_cols = [c for c in lab_df.columns if str(c).lower() not in ("time","temp","conductivity")]
freq_map = {c: extract_freq(c) for c in freq_cols if extract_freq(c) is not None}
sorted_freq_cols = sorted(freq_map.keys(), key=lambda x: freq_map[x])
freqs = [freq_map[c] for c in sorted_freq_cols]

sigma_lab = lab_df["Conductivity"].values
temp_lab = lab_df["Temp"].values
temp_shifted = temp_lab - T0

# === BUILD CORRECTION MODELS PER FREQUENCY ===
freq_models = {}
temp_models = {}
noise_levels = {}

for c in sorted_freq_cols:
    phi_lab = lab_df[c].values.astype(float)

    X_sigma = sm.add_constant(sigma_lab)
    model_sigma = sm.OLS(phi_lab, X_sigma).fit()
    phi_sigma_pred = model_sigma.predict(X_sigma)

    residuals = phi_lab - phi_sigma_pred
    X_temp_poly = np.column_stack([temp_shifted, temp_shifted**2, temp_shifted**3])
    X_temp_poly = sm.add_constant(X_temp_poly)
    model_temp = sm.OLS(residuals, X_temp_poly).fit()

    final_residuals = residuals - model_temp.predict(X_temp_poly)
    noise_std = np.std(final_residuals) * NOISE_SCALE

    # Save models in dictionary
    freq_models[c] = {
        "const": float(model_sigma.params[0]), 
        "slope": float(model_sigma.params[1])
    }

    temp_models[c] = {
        "const": float(model_temp.params[0]),
        "coef1": float(model_temp.params[1]),
        "coef2": float(model_temp.params[2]),
        "coef3": float(model_temp.params[3])
    }
    noise_levels[c] = noise_std

# Save models to JSON for reuse
with open(OUT_PARAMS, "w") as f:
    json.dump({
        "T0": T0,
        "freq_models": freq_models,
        "temp_models": temp_models,
        "noise_levels": noise_levels
    }, f, indent=2)


print(f"Models saved to {OUT_PARAMS}")
