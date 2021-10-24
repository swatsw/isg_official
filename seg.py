import csv
from genea import GENEA
import random
import os
import shutil

TRAIN_WAV_DIR = "../GENEA_2020/train/audio"
TRAIN_BVH_DIR = "../GENEA_2020/train/motion"
TRAIN_TEXT_DIR = "../GENEA_2020/train/transcripts"

TEST_WAV_DIR = "../GENEA_2020/test/audio"
TEST_BVH_DIR = "../GENEA_2020/test/motion"
TEST_TEXT_DIR = "../GENEA_2020/test/transcripts"

FILELIST_DIR = "data/filelists"
TRAIN_OUT_FILELIST = "data/filelists/genea_train.txt"
DEV_OUT_FILELIST = "data/filelists/genea_dev.txt"
TEST_OUT_FILELIST = "data/filelists/genea_test.txt"

OUT_WAV_DIR = "data/wav"
OUT_BVH_DIR = "data/bvh"

# random seg params
WIN_MIN = 5.0
WIN_MAX = 11.0
STEP_MIN = WIN_MIN / 2
STEP_MAX = WIN_MAX / 2

def sample_min_max(min_val, max_val):
    return min_val + random.random() * (max_val - min_val)

# seg by over lapping window of random len [win_min, win_max], and random step_size in [step_min, step_max]
def random_seg_single(prior_start, genea_dat, episode, segment_name,
                      win_min=WIN_MIN, win_max=WIN_MAX, step_min=STEP_MIN, step_max=STEP_MAX):
    '''

    :param prior_start: prior segment start point
    :return:
    '''
    win = sample_min_max(win_min, win_max)
    step = sample_min_max(step_min, step_max)

    curr_start = prior_start + step
    true_start, utter_text = genea_dat.segment(episode, curr_start, win, segment_name, wav_out_dir=OUT_WAV_DIR, bvh_out_dir=OUT_BVH_DIR)

    return true_start, utter_text

def random_segment_all(genea_dat):
    episodes = genea_dat.episodes
    filelist_lines = []
    for episode in episodes:
        print(f"segmenting {episode} ...")
        prior_start = 0.0
        episode_dur = genea_dat.get_wav_dur(episode)
        seg_idx = 0

        while prior_start + WIN_MAX < episode_dur:
            segment_name = f"{episode}_{seg_idx}"
            prior_start, utter_text = random_seg_single(prior_start, genea_dat, episode, segment_name)
            if utter_text is not None:
                filelist_lines.append(f"{segment_name}.wav|{utter_text}\n")
                seg_idx += 1
    return filelist_lines

def write_filelist(fpath, lines):
    with open(fpath, "w") as f:
        for line in lines:
            f.write(line)

if __name__ == "__main__":
    try:
        os.mkdir(OUT_WAV_DIR)
    except:
        pass
    try:
        os.mkdir(OUT_BVH_DIR)
    except:
        pass
    try:
        os.mkdir(FILELIST_DIR)
    except:
        pass
    try:
        os.remove(TRAIN_OUT_FILELIST)
    except:
        pass
    try:
        os.remove(DEV_OUT_FILELIST)
    except:
        pass
    try:
        os.remove(TEST_OUT_FILELIST)
    except:
        pass

    train_dat = GENEA(wav_folder=TRAIN_WAV_DIR, bvh_folder=TRAIN_BVH_DIR, text_folder=TRAIN_TEXT_DIR)
    train_flist_lines = random_segment_all(train_dat)
    test_dat = GENEA(wav_folder=TEST_WAV_DIR, bvh_folder=TEST_BVH_DIR, text_folder=TEST_TEXT_DIR)
    test_flist_lines = random_segment_all(test_dat)

    val_num = int(len(test_flist_lines)/2)
    val_flist_lines = test_flist_lines[:val_num]
    test_flist_lines = test_flist_lines[val_num:]

    write_filelist(TRAIN_OUT_FILELIST, train_flist_lines)
    write_filelist(DEV_OUT_FILELIST, val_flist_lines)
    write_filelist(TEST_OUT_FILELIST, test_flist_lines)