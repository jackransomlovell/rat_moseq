import h5py
import cv2
import numpy as np  
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection          


def downsample(file, dset='frames', shape=(80,80)):
    with h5py.File(file, 'r') as f:
        time, height, width = f[dset].shape
        offset = (height - 160) // 2
        end = offset + 160

        frames = f[dset][:, offset:end, offset:end].astype('int16')
    return np.array([cv2.resize(frame, shape) for frame in frames])

def resize_rat(frame, factor):
    """
    Resize a rat image to a new size, while keeping the centre of the image.
    Args:
        frame: np.ndarray, depth data frame, shape (h, w)
        factor: float, the factor to resize the image by
    Returns:
        resized: np.ndarray, resized image, same shape as frame
    """
    # Get the non-zero pixels
    y,x = np.nonzero(frame)
    y0, y1 = y.min(), y.max()
    x0, x1 = x.min(), x.max()
    roi = frame[y0:y1, x0:x1]

    # return roi
    h, w = y1-y0, x1-x0              # height and width or roi
    scaled_roi = cv2.resize(roi, (int(w*factor),int(h*factor)), interpolation=cv2.INTER_NEAREST)
    new_h, new_w = scaled_roi.shape[:2]

    # Clear original image to white
    # Get centre of original shape, and position of top-left of ROI in output image
    cx, cy = (x0 + x1) //2, (y0 + y1)//2
    top  = cy - new_h//2
    left = cx - new_w//2
    resized = np.zeros_like(frame)
    resized[top:top+new_h, left:left+new_w] = scaled_roi

    return resized

def resize_video(video, factor):
    """
    Resize a video to a new size, while keeping the centre of the image.
    Args:
        video: np.ndarray, video, shape (t, h, w)
        factor: float, the factor to resize the image by
    Returns:
        resized: np.ndarray, resized video, same shape as video
    """
    return np.array([resize_rat(f, factor) for f in video if f.sum() > 0])

def zscore(arr):
    """
    Z-score a numpy array.
    Args:
        arr: np.ndarray, array to z-score
    Returns:
        zscored: np.ndarray, z-scored array
    """
    return (arr - arr.mean(axis=0, keepdims=True)) / arr.std(axis=0, keepdims=True)


def flatten(x):
    """
    Flatten a numpy array.
    Args:
        x: np.ndarray, array to flatten
    Returns:
        flattened: np.ndarray, flattened array
    """
    return x.reshape(len(x), -1)

def count_nonzero(frame):
    """
    Count the number of non-zero pixels in a numpy array.
    Args:
        frame: np.ndarray, array to count non-zero pixels
    Returns:
        count: int, number of non-zero pixels
    """
    y,x = np.nonzero(frame)
    y0, y1 = y.min(), y.max()
    x0, x1 = x.min(), x.max()
    # return roi
    h, w = y1-y0, x1-x0

    return w*h

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

def median_pose(frames):
    return np.median(frames, axis=0)

def mask_mouse(frames, thresh=10):
    return frames*(frames>thresh)