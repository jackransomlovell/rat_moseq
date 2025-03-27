import h5py

def load_downsampled_rat(path):
    with h5py.File(path, 'r') as f:
        downsampled = {}
        for key in f.keys():
            downsampled[key] = f[key][()]
    return downsampled