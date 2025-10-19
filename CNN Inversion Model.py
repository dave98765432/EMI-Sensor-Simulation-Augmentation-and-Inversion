import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pywt
from scipy.signal import savgol_filter
import statsmodels.api as sm
from scipy.signal import medfilt
import json
import statsmodels.api as sm

# === Global Constants ===
RAW_FEATURES = """Offset	Time	Temp	5000	5500	6000	6500	7000	7500	8000	8500	9000	9500	10000	10500	11000	11500	12000	12500	13000	13500	14000	14500	15000	15500	16000	16500	17000	17500	18000	18500	19000	19500	20000	20500	21000	21500	22000	22500	23000	23500	24000	24500	25000	25500	26000	26500	27000	27500	28000	28500	29000	29500	30000	30500	31000	31500	32000	32500	33000	33500	34000	34500	35000	35500	36000	36500	37000	37500	38000	38500	39000	39500	40000	40500	41000	41500	42000	42500	43000	43500	44000	44500	45000	45500	46000	46500	47000	47500	48000	48500	49000	49500	50000	50500	51000	51500	52000	52500	53000	53500	54000	54500	55000	55500	56000	56500	57000	57500	58000	58500	59000	59500	60000	60500	61000	61500	62000	62500	63000	63500	64000	64500	65000	65500	66000	66500	67000	67500	68000	68500	69000	69500	70000	70500	71000	71500	72000	72500	73000	73500	74000	74500	75000	75500	76000	76500	77000	77500	78000	78500	79000	79500	80000	Conductivity""".strip().split()
MANUAL_PRIORITY_FEATURES = list(map(str, range(70000, 80000, 500)))

T0 = 37.0  # reference temp for correction

# === Temperature correction ===
def correct_to_reference_temp(df, Tcol="Temp", Ccol="Conductivity", T0=37.0):
    """
    As in your original implementation: correct phase offset columns in df to reference temp T0
    using coefficients stored in a JSON file (instrument + temp correction).
    """
    json_file = r"C:\Users\David Linton\Desktop\Final Year Project\Code\phase_correction_model.json"
    with open(json_file, "r") as f:
        params = json.load(f)

    temp_models = params["temp_models"]
    freq_models = params["freq_models"]
    T0 = params.get("T0", T0)

    corrected = df.copy()
    temp = df[Tcol].values.astype(float)
    sigma = df[Ccol].values.astype(float)
    temp_shift = temp - T0

    for c in temp_models.keys():
        if c not in df.columns:
            continue
        phi = df[c].values.astype(float)

        # Instrumental gain (kept for completeness even if not applied to corrected output)
        gain_model = freq_models[c]
        gain_pred = gain_model["const"] + gain_model.get("slope", 0.0) * sigma

        # Temperature polynomial contribution
        tm = temp_models[c]
        pred_actual = (
            tm["const"]
            + tm.get("coef1", 0.0) * temp_shift
            + tm.get("coef2", 0.0) * (temp_shift**2)
            + tm.get("coef3", 0.0) * (temp_shift**3)
        )
        pred_ref = tm.get("const", 0.0)
        temp_correction = pred_actual - pred_ref

        # Apply correction (same as your original)
        corrected[c] = phi - temp_correction

    return corrected

# === Wavelet Denoise ===
def wavelet_denoise(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level or pywt.dwt_max_level(len(signal), wavelet))
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised = [coeffs[0]] + [pywt.threshold(c, uthresh, 'soft') for c in coeffs[1:]]
    reconstructed = pywt.waverec(denoised, wavelet)
    return reconstructed[:len(signal)]

# === Filtering ===
def smoothness_loss(y_pred, y_true, alpha=0.001):
    mse = nn.MSELoss()(y_pred, y_true)
    if y_pred.numel() <= 1:
        return mse
    diff = torch.mean((y_pred[1:] - y_pred[:-1])**2)
    return mse + alpha * diff

# === Feature Selection & Scaling ===
def prioritize_and_scale(df, label_col="Conductivity"):
    labels = df[label_col].values if label_col in df else None
    features = df.drop(columns=[label_col, "Time"], errors='ignore')
    # Keep your original behaviour: only MANUAL_PRIORITY_FEATURES get weight 1, others 0
    weights = np.array([1 if col in MANUAL_PRIORITY_FEATURES else 0 for col in features.columns])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features) * weights
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    if labels is not None:
        scaled_df[label_col] = labels
    return scaled_df, scaler

# === Dataset ===
class EMIDataset(Dataset):
    def __init__(self, df):
        # keep original ordering of columns
        self.feature_cols = df.drop(columns=["Conductivity"]).columns
        self.X = torch.tensor(df.drop(columns=["Conductivity"]).values, dtype=torch.float32)
        self.y = torch.tensor(df["Conductivity"].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === CNN Model ===
class EMICNN(nn.Module):
    def __init__(self, seq_len):
        """
        seq_len: number of input frequency features (e.g. 151 or 152 including Offset)
        Architecture:
         - Conv1D(1 -> 32, k=3, pad=1) -> BN -> ReLU
         - Conv1D(32 -> 64, k=3, pad=1) -> BN -> ReLU
         - MaxPool1d(2)
         - Flatten -> FC(64 * (seq_len//2) -> 128) -> BN -> ReLU -> Dropout(0.3)
         - FC(128 -> 64) -> BN -> ReLU -> Dropout(0.2)
         - FC(64 -> 1)
        This preserves the FC sizes from your original model for functional parity.
        """
        super().__init__()
        self.seq_len = seq_len
        # conv block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # compute flattened size after pooling
        pooled_len = seq_len // 2  # floor division, same as MaxPool1d default behaviour
        flat_size = 64 * pooled_len

        # fully connected head (keeps same sizes as your original)
        self.fc1 = nn.Linear(flat_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.2)

        self.out = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len) -> reshape to (batch, 1, seq_len) for conv1d
        if x.dim() == 2:
            x = x.view(x.size(0), 1, x.size(1))
        # convs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)  # (batch, 64, seq_len//2)
        x = x.view(x.size(0), -1)  # flatten

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.drop2(x)

        x = self.out(x)
        return x

# === Training ===
def train(csv_path, epochs=50, batch_size=100, lr=1e-2):
    df = pd.read_csv(csv_path).dropna()

    # Temperature correction (unchanged)
    df = correct_to_reference_temp(df)
    if "Temp" in df.columns:
        df = df.drop(columns=["Temp"])

    # Wavelet denoise on numeric frequency columns (unchanged)
    for col in df.columns:
        if col.isdigit():
            df[col] = wavelet_denoise(df[col].values)

    # Visual check (unchanged)
    if "80000" in df.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(df["80000"].values, label="Post Noise + Temp Correction 80000 Hz", linewidth=2)
        plt.title("80000 Hz After Noise + Temperature Correction (Train)")
        plt.xlabel("Sample Index")
        plt.ylabel("Phase Offset")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # scale + prioritise (unchanged)
    df, scaler = prioritize_and_scale(df)
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)
    train_loader = DataLoader(EMIDataset(train_df), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(EMIDataset(test_df), batch_size=batch_size)

    # Determine seq_len from training dataframe (number of input columns)
    feature_cols = train_df.drop(columns=['Conductivity']).shape[1]
    seq_len = feature_cols

    # instantiate CNN model (pass seq_len)
    model = EMICNN(seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            # reshape X to (batch, 1, seq_len) inside model forward, but we can keep same handling
            optimizer.zero_grad()
            preds = model(X)  # model handles reshape
            loss = smoothness_loss(preds, y, alpha=0.001)
            loss.backward()
            optimizer.step()
        # print last batch loss as before
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # evaluation (unchanged)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X)
            test_loss += smoothness_loss(preds, y).item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    return model, scaler, df.drop(columns=["Conductivity"]).columns

# === Prediction ===
def predict(model, scaler, test_path, output_path, feature_cols):
    raw = pd.read_csv(test_path)

    # 1️⃣ Noise correction first
    filtered = raw.copy()
    if "80000" in filtered.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(filtered["80000"].values, label="Post Noise + Temp Correction 80000 Hz", linewidth=2)
        plt.title("80000 Hz After Noise + Temperature Correction (Train)")
        plt.xlabel("Sample Index")
        plt.ylabel("Phase Offset")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    filtered = correct_to_reference_temp(filtered)
    if "Temp" in filtered.columns:
        filtered = filtered.drop(columns=["Temp"])

    # 2️⃣ Temperature correction second (visuals as original)
    if "80000" in filtered.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(filtered["80000"].values, label="Post Noise + Temp Correction 80000 Hz", linewidth=2)
        plt.title("80000 Hz After Noise + Temperature Correction (Train)")
        plt.xlabel("Sample Index")
        plt.ylabel("Phase Offset")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    for col in filtered.columns:
        if col.isdigit():
            filtered[col] = wavelet_denoise(filtered[col].values)

    if "80000" in filtered.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(filtered["80000"].values, label="Post Noise + Temp Correction 80000 Hz", linewidth=2)
        plt.title("80000 Hz After Noise + Temperature Correction (Train)")
        plt.xlabel("Sample Index")
        plt.ylabel("Phase Offset")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # scale using fitted scaler
    scaled_features = scaler.transform(filtered[feature_cols])
    # keep your prediction weighting (0.9 / 0.1) as in your latest predict
    weights = np.array([0.9 if col in MANUAL_PRIORITY_FEATURES else 0.1 for col in feature_cols])
    scaled_features *= weights

    X_test = torch.tensor(scaled_features, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test).squeeze().clamp(0, 3000).numpy()
        y_pred_smooth = savgol_filter(preds, window_length=50, polyorder=1)

    out = raw.copy()

    if len(y_pred_smooth) > 0:
        plt.figure(figsize=(12, 5))
        plt.plot(y_pred_smooth, label="Post Noise + Temp Correction 80000 Hz", linewidth=2)
        plt.title("80000 Hz After Noise + Temperature Correction (Train)")
        plt.xlabel("Sample Index")
        plt.ylabel("Phase Offset")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    out["Conductivity"] = y_pred_smooth
    out.to_csv(output_path, index=False)

    if "80000" in raw.columns and "80000" in filtered.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(raw["80000"].values, label="Raw 80000 Hz", alpha=0.6)
        plt.plot(filtered["80000"].values, label="Filtered + Temp Corrected 80000 Hz", linewidth=2)
        plt.title("Noise + Temp Correction on 80000 Hz (Predict)")
        plt.xlabel("Sample Index")
        plt.ylabel("Phase Offset")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === Run ===
model, scaler, features = train(r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\Train1.csv")
predict(
    model=model,
    scaler=scaler,
    test_path=r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\Sim1Test5+Tool.csv",
    output_path=r"C:\Users\David Linton\Desktop\Final Year Project\Training Data\Sim1Test5+ToolP.csv",
    feature_cols=features
)
print("end")
