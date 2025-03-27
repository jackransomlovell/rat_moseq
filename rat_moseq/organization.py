from pathlib import Path
import h5py

def check_h5(h5_path, key='frames'):
    """
    Check if the h5 file is a valid rat_moseq h5 file.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            return key in f.keys()
    except Exception as e:
        return False

def get_rat_h5s(check=False, key='frames'):
    """
    Get all the h5 files in the directory.
    """
    rat_pdir = Path('/n/groups/datta/jlove/data/rat_seq/rat_seq_paper/data')
    h5s = list(rat_pdir.glob('**/results_00.h5'))
    if check:
        h5s = list(filter(lambda x: check_h5(x, key), h5s))
    return h5s

def get_mouse_h5s(check=False, key='frames'):
    """Get all the h5 files in the directory."""
    mouse_pdir = Path('/n/groups/datta/jlove/data/rat_seq/rat_seq_paper/data/mice_control_v2/example_data_with_results')
    mouse_h5s = list(mouse_pdir.glob('aggregate_results/*.h5'))
    if check:
        mouse_h5s = list(filter(lambda x: check_h5(x, key), mouse_h5s))
    return mouse_h5s
