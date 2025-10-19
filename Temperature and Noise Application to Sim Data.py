import numpy as np
import pandas as pd
import json
import re

def apply_phase_correction(sim_file, json_file, output_file, sim_sheet="3", out_sheet="Sim_Corrected", noise_scale=0.4):
    """
    Apply phase and temperature corrections to a SIM Excel file using precomputed JSON models.

    Parameters:
    - sim_file: str, path to the SIM input Excel file
    - json_file: str, path to the JSON file with correction models
    - output_file: str, path to save the corrected SIM Excel file
    - sim_sheet: str, sheet name in SIM file containing data (default "Sim")
    - out_sheet: str, sheet name for output (default "Sim_Corrected")
    - noise_scale: float, scaling factor for added noise (default 0.8)
    """

    # Load JSON models
    with open(json_file, "r") as f:
        params = json.load(f)

    T0 = params["T0"]
    phase_models = params["freq_models"]
    temp_models = params["temp_models"]
    noise_levels = params["noise_levels"]

    # Load SIM data
    sim_df = pd.read_excel(sim_file, sheet_name=sim_sheet)

    # Map JSON frequency keys to SIM columns
    sim_freq_cols = []
    sim_col_map = {}
    for col in sim_df.columns:
        if col in ("Temp", "Conductivity"):
            continue
        col_key = re.sub(r"\D", "", str(col))
        if col_key in phase_models:
            sim_freq_cols.append(col)
            sim_col_map[col_key] = col

    if not sim_freq_cols:
        raise ValueError("No matching frequency columns found between JSON params and SIM file!")

    # Apply corrections
    corrected = sim_df.copy()
    sigma_sim = sim_df["Conductivity"].values
    temp_shifted = sim_df["Temp"].values - T0

    for f_key, col_name in sim_col_map.items():
        phi_sim = sim_df[col_name].values

        # Phase offset correction (linear in conductivity)
        intercept, slope = phase_models[f_key]["const"], phase_models[f_key]["slope"]
        delta_sigma = (intercept + slope * sigma_sim) - phi_sim

        # Temperature correction (cubic polynomial)
        t_params = temp_models[f_key]
        delta_temp = (
            t_params["const"]
            + t_params["coef1"] * temp_shifted
            + t_params["coef2"] * (temp_shifted**2)
            + t_params["coef3"] * (temp_shifted**3)
        )

        # Noise
        noise = np.random.randn(len(phi_sim)) * noise_levels[f_key] * noise_scale

        # Apply total correction
        corrected[col_name] = phi_sim + abs(delta_sigma) + delta_temp + noise

    # Save corrected SIM data
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        corrected.to_excel(writer, sheet_name=out_sheet, index=False)

    print(f"Corrected simulation saved to {output_file}")


# =======================
# Example usage function
# =======================
def run_example():
    apply_phase_correction(
        sim_file=r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\SimTest1.xlsx",
        json_file=r"C:\Users\David Linton\Desktop\Final Year Project\Code\phase_correction_model.json",
        output_file=r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\Sim1Test1+Tool.xlsx"
    )


# Call the example function if this script is run directly
if __name__ == "__main__":
    run_example()
