
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd
import numpy as np
from itertools import product
from rat_moseq.size import flatten, zscore

def compute_changepoints(frames, mdl=None, k=6, sig=4):
    """
    Compute changepoints in a video.
    Args:
        frames: np.ndarray, video, shape (t, h, w)
        mdl: sklearn.random_projection.GaussianRandomProjection, model to use for changepoint detection
    Returns:
        cp: np.ndarray, changepoints, shape (t,)
        proj_df_smooth: pd.DataFrame, smoothed changepoints, shape (t, 300)
        mdl: sklearn.random_projection.GaussianRandomProjection, model used for changepoint detection
    """
    if mdl is None:
        mdl = GaussianRandomProjection(n_components=300, random_state=0)
        proj = mdl.fit_transform(flatten(frames))
    else:
        proj = mdl.transform(flatten(frames))

    proj_df = pd.DataFrame(zscore(zscore(proj).T).T, index=np.arange(len(proj)) / 30)
    proj_df_smooth = (
        proj_df.rolling(sig * 4, win_type="gaussian", center=True).mean(std=sig).dropna()
    )
    squared_diff = np.square(proj_df_smooth.diff(k)).shift(-k // 2)
    cp = squared_diff.mean(axis="columns")

    return cp, proj_df_smooth, mdl

def get_rps(frames, n_components=25):
    mdl = GaussianRandomProjection(n_components=n_components, random_state=0)
    proj = mdl.fit_transform(flatten(frames))
    return pd.DataFrame(proj)

def exp_decay(x, tau, baseline):
    return baseline + (1 - baseline) * np.exp(-x/tau)

def mse(x, y):
    return ((x-y)**2).mean()

def interpolate(data, zero_indices):
    non_zero_indices = np.where(data != 0)[0]
    return np.interp(zero_indices, non_zero_indices, data[non_zero_indices])

def mask_frame(frames, thresh=10):
    return frames*(frames > thresh)

def compute_rps_autocorr(rps, nframes=125):
    
    ac_dfs = []
    for r in rps:
        ac = [rps.iloc[:, r].autocorr(f) for f in range(nframes)]
        ac = np.array(ac)

        first = ac[0]
        if first < 0:
            ac = -1*ac

        percent0 = (ac == 0).sum() / ac.size
        if percent0 > .5:
            continue

        percentna = np.isnan(ac).sum() / ac.size
        if percentna > .5:
            continue

        # interpolate 0s
        zero_indices = np.where(ac == 0)[0]

        # Interpolate zero values
        ac[zero_indices] = interpolate(ac, zero_indices)

        ac_dfs.append(
            pd.DataFrame(
                dict(
                    ac=ac,
                    time=(np.arange(nframes)/30)*1000,
                    dim=r
                )
            )
        )
    ac_dfs = pd.concat(ac_dfs)
    return ac_dfs

def compute_tau_df(data, taus, baselines, nframes=125, loss=mse):
    
    x = (np.arange(nframes)/30)*1000
    dfs = []
    for i, (tau, baseline) in enumerate(product(taus, baselines)):
        ts = exp_decay(x, tau, baseline)
    
        dfs.append(
            pd.DataFrame(
                dict(
                    tau=tau,
                    baseline=baseline,
                    result=loss(ts, data)
                ),
                index=[i]
            )
        )
    dfs = pd.concat(dfs)
    return dfs