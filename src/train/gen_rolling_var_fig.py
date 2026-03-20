"""Generate rolling variance CSV from real CSI data for LaTeX pgfplots figure."""
import numpy as np
import pandas as pd
import os, json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats_results')
PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'newpaper2026')

# Load a short segment of Office HAR "work" activity
csv_path = os.path.join(DATA_DIR, 'office_har_data', 'work.csv')
df = pd.read_csv(csv_path, nrows=5000)

# CSI data is in the 'data' column as a JSON-like list string
def parse_csi_row(data_str):
    vals = json.loads(data_str)
    return np.array(vals, dtype=np.float32)

csi_matrix = np.stack(df['data'].apply(parse_csi_row).values)
print(f"CSI matrix shape: {csi_matrix.shape}")  # (N, 128)

# Parse complex CSI: pairs of (imag, real) -> amplitude
n_sub = 64
imag = csi_matrix[:, 0::2][:, :n_sub]
real = csi_matrix[:, 1::2][:, :n_sub]
amp = np.sqrt(real**2 + imag**2)

# Apply subcarrier mask (keep indices 6-31, 33-58)
mask = np.zeros(64, dtype=bool)
mask[6:32] = True   # negative freq
mask[32] = False     # DC
mask[33:59] = True   # positive freq
amp_masked = amp[:, mask]  # shape: (N, 52)

# Pick subcarrier index 10 (arbitrary mid-band)
sc = 10
x = amp_masked[:, sc]

def rolling_variance(arr, w):
    n = len(arr)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr**2)
    cs = np.concatenate([[0], cs])
    cs2 = np.concatenate([[0], cs2])
    hi = np.arange(1, n + 1)
    lo = np.clip(hi - w, 0, None)
    counts = (hi - lo).astype(float)
    means = (cs[hi] - cs[lo]) / counts
    mean_sq = (cs2[hi] - cs2[lo]) / counts
    var = np.clip(mean_sq - means**2, 0, None)
    return var

# Compute rolling variances
rv20 = rolling_variance(x, 20)
rv200 = rolling_variance(x, 200)
rv2000 = rolling_variance(x, 2000)

# Downsample for pgfplots (every 5th sample -> ~1000 points)
step = 5
indices = np.arange(0, len(x), step)

out = pd.DataFrame({
    'n': indices,
    'rv20': rv20[indices],
    'rv200': rv200[indices],
    'rv2000': rv2000[indices],
})

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, 'rolling_var_data.csv')
out.to_csv(out_path, index=False, float_format='%.4f')
print(f"Wrote {len(out)} rows to {out_path}")

# Also copy to paper directory for pgfplots
paper_path = os.path.join(PAPER_DIR, 'rolling_var_data.csv')
out.to_csv(paper_path, index=False, float_format='%.4f')
print(f"Also copied to {paper_path}")
print(f"rv20  range: [{rv20.min():.2f}, {rv20.max():.2f}]")
print(f"rv200 range: [{rv200.min():.2f}, {rv200.max():.2f}]")
print(f"rv2000 range: [{rv2000.min():.2f}, {rv2000.max():.2f}]")
