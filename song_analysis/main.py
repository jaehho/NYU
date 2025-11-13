import os, glob, json, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import umap, hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------- config ----------
AUDIO_DIR = "data/"          # folder of WAVs
SR = 30000
BAND = (1000, 10000)
N_MELS = 64
N_FFT = 1024
HOP = 320
MIN_DUR = 0.05
MAX_DUR = 2.0
GAP_MERGE_S = 0.06            # merge gaps shorter than this
UMAP_K = 15                   # neighbors
UMAP_MIN_DIST = 0.1
HDB_MIN_CLUSTER = 15          # tune 8–50 depending on dataset size
HDB_MIN_SAMPLES = None        # default = min_cluster_size
RANDOM = 0

# ---------- helpers ----------
def bandpass(y, sr, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)

def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = bandpass(y, sr, *BAND)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr

def logmel(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=BAND[0], fmax=BAND[1])
    LS = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return LS

def segment_flux(LS, hop_s):
    flux = np.diff(np.maximum(LS, -80.0), axis=1).clip(min=0).mean(axis=0)
    thr = flux.mean() + 1.0*flux.std()
    on = np.where(flux > thr)[0]
    if not on.size: return []
    # group contiguous frames
    events = []
    s = on[0]
    for i in range(1, len(on)):
        if on[i] != on[i-1]+1:
            events.append((s, on[i-1]))
            s = on[i]
    events.append((s, on[-1]))
    # merge small gaps
    merged = []
    for a,b in events:
        if merged and (a - merged[-1][1]) * hop_s <= GAP_MERGE_S:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a,b))
    # filter by duration
    segs = []
    for a,b in merged:
        t0, t1 = a*hop_s, b*hop_s
        if MIN_DUR <= (t1 - t0) <= MAX_DUR:
            segs.append((t0, t1))
    return segs

def seg_feats(LS, D1, D2, a_s, b_s, hop_s):
    aa, bb = max(0, int(a_s/hop_s)), int(b_s/hop_s)
    if bb - aa < 2:  # too few frames; skip
        return None
    M = LS[:, aa:bb]
    mu = M.mean(axis=1)
    sd = M.std(axis=1)
    d1 = D1[:, aa:bb].mean(axis=1)
    d2 = D2[:, aa:bb].mean(axis=1)
    return np.r_[mu, sd, d1, d2]


def montage(LS, spans, path):
    plt.figure(figsize=(10,3))
    librosa.display.specshow(LS, sr=SR, hop_length=HOP, x_axis='time', y_axis='mel',
                             fmin=BAND[0], fmax=BAND[1])
    for (a,b) in spans:
        plt.axvspan(a, b, alpha=0.2)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ---------- pass 1: segment and extract features ----------
rows = []
features = []
for flac in glob.glob(os.path.join(AUDIO_DIR, "*.flac")):
    y, sr = load_audio(flac)
    LS = logmel(y, sr)
    D1 = librosa.feature.delta(LS, order=1, width=9, mode="interp")
    D2 = librosa.feature.delta(LS, order=2, width=9, mode="interp")
    hop_s = HOP/sr
    segs = segment_flux(LS, hop_s)
    for (a,b) in segs:
        feat = seg_feats(LS, D1, D2, a, b, hop_s)
        if feat is None:
            continue
        features.append(feat)
        # DELETE the broken line: rows.append({...})
        rows.append({
            "file": os.path.basename(flac),
            "path": os.path.abspath(flac),   # new: keep full path
            "t_start": a,
            "t_end": b
        })
    os.makedirs("out/qa", exist_ok=True)
    montage(LS, segs, f"out/qa/{os.path.basename(flac)}.png")

X = np.vstack(features) if features else np.empty((0, 4*N_MELS))
df = pd.DataFrame(rows)
if len(X) == 0:
    raise SystemExit("No segments detected. Adjust thresholds or band.")

# ---------- scale + embed ----------
sc = StandardScaler().fit(X)
Xz = sc.transform(X)
um = umap.UMAP(n_neighbors=UMAP_K, min_dist=UMAP_MIN_DIST,
               metric="euclidean", random_state=RANDOM)
Z = um.fit_transform(Xz)

# ---------- cluster ----------
clusterer = hdbscan.HDBSCAN(min_cluster_size=HDB_MIN_CLUSTER,
                            min_samples=HDB_MIN_SAMPLES,
                            metric="euclidean", cluster_selection_epsilon=0.0,
                            prediction_data=True)
labels = clusterer.fit_predict(Z)
probs = clusterer.probabilities_

df["cluster"] = labels
df["prob"] = probs
df["x"] = Z[:,0]
df["y"] = Z[:,1]

# ---------- diagnostics ----------
os.makedirs("out", exist_ok=True)
df.to_csv("out/calltype_clusters.csv", index=False)

summary = (df[df.cluster!=-1]
           .groupby("cluster")
           .agg(n=("cluster","size"), mean_prob=("prob","mean"))
           .sort_values("n", ascending=False))
summary.to_csv("out/cluster_summary.csv")

# exemplar per cluster
exemplars = []
for k, g in df[df.cluster!=-1].groupby("cluster"):
    i = g["prob"].idxmax()
    exemplars.append(df.loc[i].to_dict())
pd.DataFrame(exemplars).to_csv("out/cluster_exemplars.csv", index=False)

# scatter
plt.figure(figsize=(6,5))
plt.scatter(df.x, df.y, s=8, c=df.cluster, cmap="tab20", alpha=0.9)
plt.title("UMAP + HDBSCAN call-type clusters")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.tight_layout()
plt.savefig("out/umap_clusters.png", dpi=160); plt.close()

print("Saved:")
print(" - out/calltype_clusters.csv")
print(" - out/cluster_summary.csv")
print(" - out/cluster_exemplars.csv")
print(" - out/umap_clusters.png")
print(" - out/qa/*.png")

# -------- per-cluster audio review export --------
# config
REVIEW_DIR = "out/audio_review"
PAD_S = 0.10                  # context on each side
MAX_PER_CLUSTER = 120         # cap per cluster
INCLUDE_NOISE = False         # export cluster -1 if True
TARGET_SR = 30000             # resample uniformly
BIT_DEPTH = "PCM_16"

os.makedirs(REVIEW_DIR, exist_ok=True)
rng = np.random.default_rng(7)

def safe_slice(path, t0, t1, pad, target_sr):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    a = max(0, int((t0 - pad) * sr))
    b = min(len(y), int((t1 + pad) * sr))
    return y[a:b], sr

# manifest rows
mrows = []

# choose which clusters to export
clusters = sorted(df.cluster.unique())
if not INCLUDE_NOISE:
    clusters = [c for c in clusters if c != -1]

# later: per-cluster audio review — use 'path' if available
for c in clusters:
    g = df[df.cluster == c].copy().sort_values("prob", ascending=False)
    head_n = min(20, len(g))
    pick = list(g.head(head_n).index)
    rest = [i for i in g.index if i not in pick]
    if len(pick) < MAX_PER_CLUSTER:
        pick += list(rng.choice(rest, size=min(MAX_PER_CLUSTER - len(pick), len(rest)), replace=False))
    g = df.loc[pick]

    cdir = os.path.join(REVIEW_DIR, f"cluster_{c:02d}")
    os.makedirs(cdir, exist_ok=True)

    for _, row in g.iterrows():
        src_path = row.get("path", os.path.join(AUDIO_DIR, row["file"]))  # fallback if 'path' absent
        yseg, sr = safe_slice(src_path, row.t_start, row.t_end, PAD_S, TARGET_SR)
        fout = os.path.join(
            cdir,
            f"{Path(row['file']).stem}_{row.t_start:07.3f}-{row.t_end:07.3f}_p{row.prob:.2f}.wav"
        )
        sf.write(fout, yseg, sr, subtype=BIT_DEPTH)
        mrows.append({
            "cluster": c,
            "file": row["file"],
            "t_start": row.t_start,
            "t_end": row.t_end,
            "prob": round(float(row.prob), 4),
            "export_path": fout
        })

# write manifest and simple HTML index
manifest = pd.DataFrame(mrows)
manifest.to_csv(os.path.join(REVIEW_DIR, "manifest.csv"), index=False)

html = ["<html><head><meta charset='utf-8'><title>Call-type review</title></head><body>"]
for c in clusters:
    html.append(f"<h2>Cluster {c}</h2>")
    sub = manifest[manifest.cluster == c]
    for _, r in sub.iterrows():
        rel = os.path.relpath(r.export_path, REVIEW_DIR)
        html.append(f"<div><audio controls src='{rel}'></audio> "
                    f"<code>{r['file']} [{r['t_start']:.3f}-{r['t_end']:.3f}] p={r['prob']:.2f}</code></div>")
html.append("</body></html>")
with open(os.path.join(REVIEW_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write("\n".join(html))

print("Per-cluster audio review saved to:", REVIEW_DIR)
print(" - manifest.csv")
print(" - index.html (open in a browser)")
