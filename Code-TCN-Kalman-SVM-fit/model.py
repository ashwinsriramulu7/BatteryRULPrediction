import os, math, random, json, warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import pwlf
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ==== CONFIG ====
CHARGE_CSV = "../B0005_charge.csv"
DISCHARGE_CSV = "../B0005_discharge.csv"
OUT_DIR = "out_sota_smooth"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

SEQ_LEN = 512
CHANNELS = 3
LR = 4e-4
BATCH_SIZE = 16
EPOCHS = 80
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOL_AH = 1.4
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==== UTIL ====
def rmse(a,b): return math.sqrt(mean_squared_error(a,b))
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", name), dpi=160)
    plt.close(fig)

# ==== LOAD ====
def load_data():
    ch = pd.read_csv(CHARGE_CSV)
    dis = pd.read_csv(DISCHARGE_CSV)
    return ch, dis

# ==== KALMAN FILTER + SMOOTHING ====
def apply_kalman(series: np.ndarray, var_scale: float=1e-3) -> np.ndarray:
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series[0],
        initial_state_covariance=1,
        observation_covariance=var_scale,
        transition_covariance=var_scale)
    smoothed_state, _ = kf.smooth(series)
    return smoothed_state.flatten()

def smooth_cycle_data(df_cycle: pd.DataFrame) -> pd.DataFrame:
    df = df_cycle.copy()
    for col in ["voltage", "current", "temperature"]:
        arr = df[col].values
        if len(arr) > 7:
            arr = savgol_filter(arr, window_length=7, polyorder=2)
        arr = apply_kalman(arr, var_scale=1e-4)
        df[col] = arr
    return df

def resample_cycle(subdf: pd.DataFrame, seq_len: int=SEQ_LEN) -> np.ndarray:
    subdf = subdf.sort_values("time_s")
    t = subdf["time_s"].values
    grid = np.linspace(t.min(), t.max(), seq_len)
    def interp(col): return np.interp(grid, t, subdf[col].values)
    v = interp("voltage"); i = interp("current"); temp = interp("temperature")
    return np.stack([v,i,temp]).astype(np.float32)

def compute_feats(subdf: pd.DataFrame) -> Dict[str,float]:
    feats = {}
    for col in ["voltage","current","temperature"]:
        arr = subdf[col].values
        feats[f"{col}_mean"] = np.mean(arr)
        feats[f"{col}_std"] = np.std(arr)
        feats[f"{col}_min"] = np.min(arr)
        feats[f"{col}_max"] = np.max(arr)
    t = subdf["time_s"].values
    feats["duration"] = t.max()-t.min() if len(t)>0 else 0
    if len(t)>2:
        dvdt = np.diff(subdf["voltage"])/np.diff(t)
        feats["dvdt_mean"] = np.mean(dvdt)
        feats["dvdt_std"] = np.std(dvdt)
    else:
        feats["dvdt_mean"]=feats["dvdt_std"]=0
    feats["last_v"]=subdf["voltage"].values[-1]
    feats["last_i"]=subdf["current"].values[-1]
    feats["last_t"]=subdf["temperature"].values[-1]
    return feats

# ==== BUILD DATASET ====
def build_dataset(dis: pd.DataFrame):
    caps = dis.groupby("cycle_number")["capacity_ahr"].max().reset_index()
    seqs, feats, ys, cyc = [], [], [], []
    for _, row in caps.iterrows():
        cno = int(row["cycle_number"])
        sub = dis[dis["cycle_number"]==cno]
        if len(sub)<3: continue
        sub = smooth_cycle_data(sub)
        seqs.append(resample_cycle(sub))
        feats.append(compute_feats(sub))
        ys.append(row["capacity_ahr"])
        cyc.append(cno)
    X_seq = np.stack(seqs)
    X_feat = pd.DataFrame(feats)
    y = np.array(ys)
    return X_seq, X_feat, y, np.array(cyc)

# ==== MODEL ====
class Chomp1d(nn.Module):
    def __init__(self, c): super().__init__(); self.c=c
    def forward(self,x): return x[:,:,:-self.c]

class TemporalBlock(nn.Module):
    def __init__(self, in_c,out_c,ks,dl,drop=0.1):
        super().__init__()
        pad=(ks-1)*dl
        self.conv1=nn.Conv1d(in_c,out_c,ks,padding=pad,dilation=dl)
        self.chomp1=Chomp1d(pad)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(out_c)
        self.drop=nn.Dropout(drop)
        self.res=nn.Conv1d(in_c,out_c,1) if in_c!=out_c else None
    def forward(self,x):
        out=self.drop(self.relu(self.bn(self.chomp1(self.conv1(x)))))
        res=x if self.res is None else self.res(x)
        return torch.relu(out+res)

class TCNEncoder(nn.Module):
    def __init__(self,in_c=3,chs=[32,64,128]):
        super().__init__()
        layers=[]
        for i,c in enumerate(chs):
            ic=in_c if i==0 else chs[i-1]
            layers.append(TemporalBlock(ic,c,3,2**i))
        self.net=nn.Sequential(*layers)
        self.pool=nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        y=self.net(x)
        return self.pool(y).squeeze(-1)

class FusionRegressor(nn.Module):
    def __init__(self,feat_dim,tcn_out):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(feat_dim,128),nn.ReLU(),nn.BatchNorm1d(128),
            nn.Linear(128,64),nn.ReLU(),nn.BatchNorm1d(64))
        self.fc=nn.Sequential(
            nn.Linear(64+tcn_out,128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,1))
    def forward(self,tcn,feat):
        m=self.mlp(feat)
        x=torch.cat([tcn,m],1)
        return self.fc(x).squeeze(-1)

# ==== DATASET WRAPPER ====
class BatDS(Dataset):
    def __init__(self,Xs,Xf,y):
        self.Xs=torch.tensor(Xs).float()
        self.Xf=torch.tensor(Xf).float()
        self.y=torch.tensor(y).float()
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.Xs[i],self.Xf[i],self.y[i]

# ==== TRAIN ====
def train_epoch(tcn,fuse,loader,opt,crit):
    tcn.train(); fuse.train(); loss_sum=0;n=0
    for xs,xf,y in loader:
        xs,xf,y=xs.to(DEVICE),xf.to(DEVICE),y.to(DEVICE)
        opt.zero_grad()
        out=fuse(tcn(xs),xf)
        loss=crit(out,y)
        loss.backward(); opt.step()
        loss_sum+=loss.item()*len(y); n+=len(y)
    return loss_sum/n

def eval_epoch(tcn,fuse,loader,crit):
    tcn.eval(); fuse.eval(); loss_sum=0;n=0;preds=[];trues=[]
    with torch.no_grad():
        for xs,xf,y in loader:
            xs,xf,y=xs.to(DEVICE),xf.to(DEVICE),y.to(DEVICE)
            out=fuse(tcn(xs),xf)
            loss=crit(out,y)
            loss_sum+=loss.item()*len(y); n+=len(y)
            preds.append(out.cpu().numpy()); trues.append(y.cpu().numpy())
    return loss_sum/n, np.concatenate(preds), np.concatenate(trues)

# ==== MAIN ====
def main():
    print("Loading...")
    _, dis = load_data()
    print("Building dataset (w/ smoothing)...")
    X_seq, X_feat, y_cap, cyc = build_dataset(dis)
    print(f"{len(y_cap)} cycles processed")

    # smooth target
    y_cap_smooth = savgol_filter(y_cap, window_length=5, polyorder=2)
    C0 = y_cap_smooth[0]
    y_soh = y_cap_smooth/C0
    y_mean, y_std = np.mean(y_soh), np.std(y_soh)
    y_norm = (y_soh - y_mean)/y_std

    # scalers
    seq_scaler = StandardScaler().fit(X_seq.transpose(0,2,1).reshape(-1,CHANNELS))
    feat_scaler = StandardScaler().fit(X_feat.values)
    X_seq_scaled = seq_scaler.transform(X_seq.transpose(0,2,1).reshape(-1,CHANNELS)).reshape(X_seq.shape[0],SEQ_LEN,CHANNELS).transpose(0,2,1)
    X_feat_scaled = feat_scaler.transform(X_feat.values)

    # chronological split
    N=len(y_norm); n_train=int(0.7*N); n_val=int(0.15*N)
    idx_train=np.arange(0,n_train); idx_val=np.arange(n_train,n_train+n_val); idx_test=np.arange(n_train+n_val,N)
    ds_train=BatDS(X_seq_scaled[idx_train],X_feat_scaled[idx_train],y_norm[idx_train])
    ds_val=BatDS(X_seq_scaled[idx_val],X_feat_scaled[idx_val],y_norm[idx_val])
    ds_test=BatDS(X_seq_scaled[idx_test],X_feat_scaled[idx_test],y_norm[idx_test])
    ld_train=DataLoader(ds_train,BATCH_SIZE,shuffle=True)
    ld_val=DataLoader(ds_val,BATCH_SIZE)
    ld_test=DataLoader(ds_test,BATCH_SIZE)

    # models
    tcn=TCNEncoder().to(DEVICE)
    fuse=FusionRegressor(X_feat.shape[1],128).to(DEVICE)
    opt=torch.optim.AdamW(list(tcn.parameters())+list(fuse.parameters()),lr=LR)
    crit=nn.SmoothL1Loss()
    best=float("inf"); patience=0; hist={"train":[], "val":[]}
    print("Training...")
    for ep in range(1,EPOCHS+1):
        tr=train_epoch(tcn,fuse,ld_train,opt,crit)
        vl,_,_=eval_epoch(tcn,fuse,ld_val,crit)
        hist["train"].append(tr); hist["val"].append(vl)
        if vl<best: best=vl; patience=0; torch.save({"tcn":tcn.state_dict(),"fuse":fuse.state_dict()},os.path.join(OUT_DIR,"best.pth"))
        else:
            patience+=1
            if patience>=PATIENCE: print("Early stop."); break
        if ep%2==0: print(f"Epoch {ep:03d} | Train {tr:.4f} | Val {vl:.4f}")
    ck=torch.load(os.path.join(OUT_DIR,"best.pth")); tcn.load_state_dict(ck["tcn"]); fuse.load_state_dict(ck["fuse"])
    tcn.eval(); fuse.eval()

    # prediction full
    all_loader=DataLoader(BatDS(X_seq_scaled,X_feat_scaled,np.zeros_like(y_norm)),64)
    preds=[]
    with torch.no_grad():
        for xs,xf,_ in all_loader:
            xs,xf=xs.to(DEVICE),xf.to(DEVICE)
            preds.append(fuse(tcn(xs),xf).cpu().numpy())
    preds=np.concatenate(preds)
    preds_cap = (preds*y_std + y_mean)*C0

    # === Postprocess: 3-segment SVM regression ===
    x = np.array(cyc, dtype=float)
    y = np.array(preds_cap, dtype=float)
    plr = pwlf.PiecewiseLinFit(x, y)
    bkpts = plr.fit(3)
    segments_x = [(int(bkpts[i]), int(bkpts[i+1])) for i in range(len(bkpts)-1)]
    svr_models, sv_lines = [], []

    for (x_start, x_end) in segments_x:
        seg_mask = (x >= x_start) & (x <= x_end)
        x_seg = x[seg_mask].reshape(-1,1)
        y_seg = y[seg_mask]
        svr = SVR(kernel='rbf', C=100, epsilon=0.005)
        svr.fit(x_seg, y_seg)
        svr_models.append(svr)
        sv_x = svr.support_vectors_.flatten()
        sv_y = y_seg[svr.support_]
        coef = np.polyfit(sv_x, sv_y, 1)
        sv_lines.append(coef)

    # plot
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x, y, 'o', color='C0', alpha=0.5, label='TCN prediction')
    colors=['C1','C2','C3']
    for i,(x_start,x_end) in enumerate(segments_x):
        seg_x = np.linspace(x_start,x_end,100)
        svr_y = svr_models[i].predict(seg_x.reshape(-1,1))
        m,b = sv_lines[i]
        ax.plot(seg_x, svr_y, '-', color=colors[i], lw=2.5, label=f'SVM seg {i+1}')
        ax.plot(seg_x, m*seg_x+b, '--', color=colors[i], alpha=0.7, label=f'Avg line {i+1}')
        ax.axvline(x_start, color='k', ls=':', alpha=0.6)
    ax.axhline(EOL_AH, color='r', ls='--', label='EOL')
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("3-Segment SVM-based Degradation Modeling")
    ax.legend()
    savefig(fig, "svm_piecewise_fit.png")

    segment_info=[]
    for i,(x_start,x_end) in enumerate(segments_x):
        m,b=sv_lines[i]
        segment_info.append({
            "segment":i+1,
            "x_start":x_start,
            "x_end":x_end,
            "avg_support_slope":float(m),
            "avg_support_intercept":float(b)
        })
    with open(os.path.join(OUT_DIR,"svm_piecewise_segments.json"),"w") as f:
        json.dump({"breakpoints":bkpts.tolist(),"segments":segment_info},f,indent=2)

    print("✅ SVM-based segmentation complete.")
    print("Breakpoints:", np.round(bkpts,2))
    print("Segment slopes:", [round(s['avg_support_slope'],5) for s in segment_info])
    return cyc, preds_cap

# ==== EXTENDED EVALUATION ====
def extended_evaluation(cyc, preds_cap):
    """Run complete evaluation and plotting after main() finishes."""
    from sklearn.metrics import r2_score, mean_absolute_percentage_error

    meta_path = os.path.join(OUT_DIR, "svm_piecewise_segments.json")
    if not os.path.exists(meta_path):
        print("❌ No SVM segments file found. Skipping evaluation.")
        return

    # --- Load previously saved outputs ---
    with open(meta_path, "r") as f:
        meta = json.load(f)
    breakpoints = meta["breakpoints"]
    segment_info = meta["segments"]

    x = np.array(cyc)
    y_pred_tcn = np.array(preds_cap)

    # Load true capacities
    _, dis = load_data()
    y_true = dis.groupby("cycle_number")["capacity_ahr"].max().values[:len(x)]
    y_true = savgol_filter(y_true, window_length=5, polyorder=2)

    # ======================
    # 1️⃣ TCN-Only Evaluation
    # ======================
    print("\n[TCN MODEL EVALUATION]")
    mae_tcn = mean_absolute_error(y_true, y_pred_tcn)
    rmse_tcn = rmse(y_true, y_pred_tcn)
    r2_tcn = r2_score(y_true, y_pred_tcn)
    print(f"MAE={mae_tcn:.4f} | RMSE={rmse_tcn:.4f} | R²={r2_tcn:.4f}")

    # Compute derived RUL
    rul_true = np.maximum(0, (y_true - EOL_AH) / np.abs(np.gradient(y_true)))
    rul_pred = np.maximum(0, (y_pred_tcn - EOL_AH) / np.abs(np.gradient(y_pred_tcn)))

    # --- 10 key TCN plots ---
    resid = y_pred_tcn - y_true

    # 1. True vs Predicted
    fig = plt.figure()
    plt.plot(x, y_true, 'o', label='True', alpha=0.6)
    plt.plot(x, y_pred_tcn, '-', label='TCN Pred')
    plt.axhline(EOL_AH, color='k', ls='--')
    plt.legend(); plt.xlabel("Cycle"); plt.ylabel("Capacity (Ah)")
    plt.title("TCN: True vs Predicted Capacity")
    savefig(fig, "tcn_cap_true_vs_pred.png")

    # 2. Residual histogram
    fig = plt.figure(); plt.hist(resid, bins=30, alpha=0.6)
    plt.title("TCN: Residual Distribution"); plt.xlabel("Pred - True")
    savefig(fig, "tcn_residual_hist.png")

    # 3. Residual vs Cycle
    fig = plt.figure(); plt.plot(x, resid, '.', alpha=0.7)
    plt.axhline(0, color='k', ls='--')
    plt.title("TCN: Residuals vs Cycle"); plt.xlabel("Cycle")
    savefig(fig, "tcn_residual_vs_cycle.png")

    # 4. True vs Pred scatter
    fig = plt.figure(); plt.scatter(y_true, y_pred_tcn, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.title("TCN: True vs Predicted Scatter"); plt.xlabel("True"); plt.ylabel("Pred")
    savefig(fig, "tcn_scatter_true_vs_pred.png")

    # 5. RUL vs Cycle
    fig = plt.figure(); plt.plot(x, rul_true, label='True')
    plt.plot(x, rul_pred, label='Pred'); plt.legend()
    plt.title("TCN: Derived RUL vs Cycle")
    savefig(fig, "tcn_rul_curve.png")

    # 6. Rolling MAE
    window = 10
    rolling_mae = np.convolve(np.abs(resid), np.ones(window)/window, mode='valid')
    fig = plt.figure(); plt.plot(x[:len(rolling_mae)], rolling_mae)
    plt.title("TCN: Rolling MAE (window=10)")
    savefig(fig, "tcn_rolling_mae.png")

    # 7. Degradation Rate
    fig = plt.figure()
    plt.plot(x, np.gradient(y_pred_tcn), label="TCN rate")
    plt.plot(x, np.gradient(y_true), label="True rate")
    plt.legend(); plt.title("TCN: Degradation Rate")
    savefig(fig, "tcn_degradation_rate.png")


    # 8. Normalized trends
    fig = plt.figure()
    plt.plot(x, (y_true - y_true.min())/(y_true.max()-y_true.min()), label="True")
    plt.plot(x, (y_pred_tcn - y_pred_tcn.min())/(y_pred_tcn.max()-y_pred_tcn.min()), label="Pred")
    plt.legend(); plt.title("TCN: Normalized Degradation Curves")
    savefig(fig, "tcn_normalized_trend.png")

    # 9. Residual quantiles
    fig = plt.figure()
    plt.plot(np.quantile(np.abs(resid), [0.1, 0.25, 0.5, 0.75, 0.9]), 'o-')
    plt.title("TCN: Residual Quantiles")
    savefig(fig, "tcn_error_quantiles.png")

    # 10. Error heatmap
    fig = plt.figure()
    plt.scatter(x, resid, c=y_true, cmap='coolwarm', s=25)
    plt.colorbar(label='True Capacity')
    plt.title("TCN: Residual Heatmap")
    savefig(fig, "tcn_error_heatmap.png")

    # ======================
    # 2️⃣ SVM Segment Evaluation
    # ======================
    print("\n[SVM SEGMENT EVALUATION]")
    svm_fit = np.zeros_like(y_pred_tcn)
    for seg in segment_info:
        mask = (x >= seg["x_start"]) & (x <= seg["x_end"])
        m, b = seg["avg_support_slope"], seg["avg_support_intercept"]
        svm_fit[mask] = m * x[mask] + b

    mae_svm = mean_absolute_error(y_true, svm_fit)
    rmse_svm = rmse(y_true, svm_fit)
    mape_svm = mean_absolute_percentage_error(y_true, svm_fit)
    r2_svm = r2_score(y_true, svm_fit)
    print(f"MAE={mae_svm:.4f} | RMSE={rmse_svm:.4f} | MAPE={mape_svm:.4f} | R²={r2_svm:.4f}")

    # 1. Combined plot
    fig = plt.figure()
    plt.plot(x, y_pred_tcn, label='TCN', alpha=0.5)
    plt.plot(x, svm_fit, label='SVM 3-Seg', linewidth=2)
    for b in breakpoints[1:-1]:
        plt.axvline(b, color='k', ls='--', alpha=0.7)
    plt.legend(); plt.title("SVM Segmented Fit vs TCN")
    savefig(fig, "svm_segment_fit.png")

    # ======================
    # 3️⃣ Comparative Summary
    # ======================
    print("\n[COMPARATIVE ANALYSIS]")
    fig = plt.figure()
    plt.plot(x, y_true, label="True", lw=2)
    plt.plot(x, y_pred_tcn, label="TCN", lw=1.5)
    plt.plot(x, svm_fit, label="SVM 3-Seg", lw=2)
    plt.legend(); plt.title("True vs TCN vs SVM")
    savefig(fig, "compare_all_overlay.png")

    print("\n✅ Extended evaluation and all plots saved under:", os.path.join(OUT_DIR, "figures"))


if __name__ == "__main__":
    cyc, preds_cap = main()
    extended_evaluation(cyc, preds_cap)
