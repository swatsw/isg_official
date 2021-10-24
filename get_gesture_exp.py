import os
import sys

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib as jl

from pymo.writers import BVHWriter
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline

import math
from numpy.lib.stride_tricks import sliding_window_view
import pdb
from scipy.stats import norm
from scipy.signal import savgol_filter

BVH_DIR = "data/bvh"
FLISTS = {
    'train': 'data/filelists/genea_train.txt',
    'dev': 'data/filelists/genea_dev.txt',
    'test': 'data/filelists/genea_dev.txt'
}

GESTURE_JOINTS = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm',
             'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

def extract_joint_angles(bvh_dir, files, destpath, fps=60, set_name=""):
    p = BVHParser()

    data_all = list()
    print("Importing data...")
    for f in files:
        ff = os.path.join(bvh_dir, f + '.bvh')
        print(ff)
        data_all.append(p.parse(ff))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('jtsel', JointSelector(
            GESTURE_JOINTS,
            include_root=True)),
        ('exp', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())
    ])

    print("Processing...")
    out_data = data_pipe.fit_transform(data_all)
    jl.dump(data_pipe, os.path.join(destpath, f'data_pipe-{set_name}.sav'))

    # optional saving
    # fi = 0
    # for f in files:
    #     ff = os.path.join(destpath, f)
    #     print(ff)
    #     np.savez(ff + ".npz", clips=out_data[fi])
    #     fi = fi + 1
    return out_data

def fit_and_standardize(data):
    # shape = data.shape
    flat = np.concatenate(data, axis=0)
    # flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = [scaler.transform(x) for x in data]
    return scaled, scaler

def standardize(data, scaler):
    scaled = [scaler.transform(x) for x in data]
    return scaled

def load_scaler(fpath='std_exp_scaler.sav'):
    assert os.path.isfile(fpath), "specified scaler file does not exist"
    return jl.load(fpath)

def load_data_pipeline(fpath):
    assert os.path.isfile(fpath), "specified data_pipe file does not exist"
    # fpath = f'{dir}/{fname}'
    return jl.load(fpath)

# def reverse_standardize(data, scaler):
#     unscaled = [scaler.inverse_transform(x) for x in data]
#     return unscaled
def mel_resample(bvh_arr, hop_length=256, sampling_rate=22050, bvh_fps=60):
    audio_frame_hop_time = hop_length / sampling_rate
    bvh_frame_time = 1.0 / bvh_fps
    total_bvh_time = bvh_arr.shape[0] * audio_frame_hop_time
    num_out_frame = math.floor(total_bvh_time / bvh_frame_time)
    align_indices = np.arange(num_out_frame, dtype=np.float32)
    align_indices *= bvh_frame_time
    align_indices /= audio_frame_hop_time
    align_indices = np.rint(align_indices).astype(np.int)
    out_bvh = bvh_arr[align_indices, :]
    return out_bvh

SMOOTHING_METHODS = ["box", "normal", "savgol"]
NORMAL_STEP_SIZE = 1
def smoothing_arr(arr, half_window_size: int, method):
    assert method in SMOOTHING_METHODS

    arr = np.pad(arr, ((half_window_size,), (0,)), 'edge')
    window_size = half_window_size*2+1
    if method == "box":
        arr = sliding_window_view(arr, window_shape=window_size, axis=0)
        box_filter = np.ones((window_size, 1))
        arr = np.matmul(arr, box_filter) / window_size
        arr = arr.squeeze(-1)
    if method == "normal":
        arr = sliding_window_view(arr, window_shape=window_size, axis=0)
        normal_filter_steps = np.arange(window_size) - half_window_size
        normal_filter_steps = normal_filter_steps * NORMAL_STEP_SIZE
        normal_filter = norm.pdf(normal_filter_steps)
        normal_filter = np.expand_dims(normal_filter, -1)
        arr = np.matmul(arr, normal_filter) / normal_filter.sum()
        arr = arr.squeeze(-1)
    if method == "savgol":
        arr = savgol_filter(arr, window_length=window_size, polyorder=2)
    return arr

def std_exp_to_bvh(exp_arr, out_bvh_fpath, out_fps=60, smoothing='normal', smoothing_half_ws=3,
                   scaler_fpath=f'{BVH_DIR}/std_exp_scaler.sav',
                   data_pipe_fpath=f'{BVH_DIR}/data_pipe-train.sav'):
    # flist = os.listdir(dir)
    # flist = [x for x in flist if model_name in x and x.endswith(npf_suffix)]#x.endswith("gesture.npy")]
    assert scaler_fpath, "must specify fpath for saved scaler"
    scaler = load_scaler(scaler_fpath)

    assert data_pipe_fpath, "must specify fpath for saved data pipe"
    data_pipeline = load_data_pipeline(data_pipe_fpath)

    if len(exp_arr.shape) == 3:
        exp_arr = exp_arr[0].T
    exp_arr = mel_resample(exp_arr, bvh_fps=out_fps)

    if smoothing is not None:
        exp_arr = smoothing_arr(exp_arr, half_window_size=smoothing_half_ws, method=smoothing)
    exp_arr = scaler.inverse_transform(exp_arr)

    gesture_bvh = data_pipeline.inverse_transform([exp_arr])[0]

    # pdb.set_trace()
    bvh_writer = BVHWriter()
    gesture_bvh.framerate = 1 / out_fps
    bvh_writer.write(gesture_bvh, open(out_bvh_fpath, "w"))

def get_gesture_exp(bvh_list, set_name="", fps=60):
    if len(bvh_list) == 0:
        return []

    return extract_joint_angles(BVH_DIR, bvh_list, BVH_DIR, fps=fps, set_name=set_name)

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

# a segment is only valid if has both audio-transcript(in filelist.txt) and bvh
def get_valid_segments():
    bvh_dir_flist = os.listdir(BVH_DIR)
    bvh_dir_flist = [x[:x.index(".bvh")] for x in bvh_dir_flist if x.endswith(".bvh")]

    segments_dict = {}
    for flist_set in FLISTS:
        flist = load_filepaths_and_text(FLISTS[flist_set])
        segment_list = [x[0] for x in flist]
        segment_list = [os.path.basename(x) for x in segment_list]
        segment_list = [os.path.splitext(x)[0] for x in segment_list]
        valid_segment_list = [x for x in segment_list if x in bvh_dir_flist]
        invalid_segment_list = [x for x in segment_list if x not in bvh_dir_flist]
        print("Invalid segments in {}: ".format(flist_set), invalid_segment_list)
        segments_dict[flist_set] = valid_segment_list
    return segments_dict

if __name__ == "__main__":
    segments_dict = get_valid_segments()
    gesture_exp_dict = {x: get_gesture_exp(segments_dict[x], set_name=x)
                        for x in segments_dict}

    out, scaler = fit_and_standardize(gesture_exp_dict["train"])
    gesture_exp_dict["train"] = out
    gesture_exp_dict["dev"] = standardize(gesture_exp_dict["dev"], scaler)
    gesture_exp_dict["test"] = standardize(gesture_exp_dict["test"], scaler)

    for k in gesture_exp_dict:
        for name, x in zip(segments_dict[k], gesture_exp_dict[k]):
            np.save(f"{BVH_DIR}/{name}_std_exp.npy", x)

    jl.dump(scaler, f'{BVH_DIR}/std_exp_scaler.sav')
