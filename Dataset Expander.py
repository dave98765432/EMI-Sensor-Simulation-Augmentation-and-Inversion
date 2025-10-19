import pandas as pd
import numpy as np

# Load your Excel file
file_path = r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\1.xlsx"
sheet3 = pd.read_excel(file_path, sheet_name="Sheet1")

# Target number of rows
target_rows = 2400

# Create new index scaled to the target size
old_index = np.linspace(0, 1, len(sheet3))
new_index = np.linspace(0, 1, target_rows)

# Interpolate all columns
expanded = pd.DataFrame(
    np.vstack([np.interp(new_index, old_index, sheet3[col]) for col in sheet3.columns]).T,
    columns=sheet3.columns
)

# Save
output_path = r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\SimTest5.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    sheet3.to_excel(writer, sheet_name="Sheet3_original", index=False)
    expanded.to_excel(writer, sheet_name="Sheet3_interpolated", index=False)

print("Saved:", output_path)