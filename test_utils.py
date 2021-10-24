import sys
sys.path.append('waveglow/')
import numpy as np
import torch
from hparams import create_hparams# from hparams import create_hparams
from train import *
import yaml

from model import Tacotron2
from layers import TacotronSTFT, STFT
from scipy.io import wavfile
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from datetime import datetime
import scipy.io.wavfile as wavfile
import time
import json
import scipy.stats
import re

def init_wavglow(waveglow_path, use_half=True, use_cpu=False):
    waveglow = torch.load(waveglow_path)['model']
    waveglow.eval()
    if not use_cpu:
        waveglow.cuda()
    if use_half:
        waveglow.half()
        for k in waveglow.convinv:
            k.float()
    denoiser = Denoiser(waveglow)
    return waveglow, denoiser

def load_test_model(model_name, iter, hparams, return_iter=False):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    ckpt_folder = f"{hparams.output_directory}/{model_name}/"
    ckpt_files = os.listdir(ckpt_folder)
    ckpt_files = [x for x in ckpt_files if "checkpoint" in x]
    ckpt_iters = [int(x.split("_")[1]) for x in ckpt_files]
    max_iter = max(ckpt_iters)
    if iter == "latest":
        iter = max_iter

    checkpoint_path = ckpt_folder + "checkpoint_" +  str(iter)
    print(f"loading [{checkpoint_path}] for testing...")
    model = load_model(hparams)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    ckpt_weight_names = list(checkpoint_dict['state_dict'].keys())
    if 'decoder.gesture_lstm_rnn.weight_ih' in ckpt_weight_names:
        for weight_name in ckpt_weight_names:
            if 'gesture_lstm_rnn' in weight_name:
                new_weight_name = ".".join(weight_name.split(".")[:-1]) + ".0." + weight_name.split(".")[-1]
                checkpoint_dict['state_dict'][new_weight_name] = checkpoint_dict['state_dict'].pop(weight_name)
    model.load_state_dict(checkpoint_dict['state_dict'])
    # model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    # _ = model.cuda().eval()#.half()
    model.cuda().eval()
    if hparams.fp16_run:
        model.half()
    if hparams.gesture_use_moglow:
        model.decoder.moglow.set_actnorm_init(inited=True)
        assert hparams.batch_size == 1
        model.decoder.moglow.z_shape[0] = 1

    if return_iter:
        return model, iter, max_iter
    else:
        return model

# def init(model_config_file, test_config_file):

def config_hparams(model_config, test_config):
    shared_hparams = model_config["shared_hparams"]
    # shared_hparams = ["{key}={value}".format(key=x, value=shared_hparams[x]) for x in shared_hparams.keys()]
    test_hparams = model_config["test_hparams"]
    # test_hparams = ["{key}={value}".format(key=x, value=test_hparams[x]) for x in test_hparams.keys()]
    test_config_hparams = test_config["hparams"]
    # test_config_hparams = ["{key}={value}".format(key=x, value=test_config_hparams[x]) for x in
    #                        test_config_hparams.keys()]
    # hparams = ",".join(shared_hparams + test_hparams + test_config_hparams)
    # hparams = create_hparams(hparams)

    hparams = create_hparams(shared_hparams, test_hparams, test_config_hparams)

    return hparams

class SynthTimer():
    def __init__(self):
        self.times = []
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.times.append(time.time() - self.start_time)

    def output_json(self, fpath):
        out = {"times": self.times}

        # stats
        times = np.array(self.times)
        out["mean"] = np.mean(times)
        out["std"] = np.std(times)

        confidence_level = 0.95
        df = len(self.times) - 1
        sample_mean = np.mean(times)
        sample_sem = scipy.stats.sem(times)
        confidence_level = scipy.stats.t.interval(confidence_level, df, sample_mean, sample_sem)
        out["confidence_interval"] = confidence_level[1] - sample_mean

        json.dump(out, open(fpath, "w"), indent=4)

def get_phon_seq(in_txt, g2p):
    txt = re.sub('[\!.?]+', '', in_txt)
    txt = re.sub(';', '.', txt)
    phon = g2p(txt)
    for j, n in enumerate(phon):
        if n == ' ':
            phon[j] = '} {'
    phon = '{ ' + ' '.join(phon) + ' }.'
    phon = re.sub(r'(\s+){ , }(\s+)', ',', phon)
    phon = re.sub(r'(\s+)?{ . }(\s+)?', ';', phon)
    # phon = re.sub(r' ; ', ';', phon)
    phon = re.sub(r'{ ', '{', phon)
    phon = re.sub(r' }', '}', phon)
    return phon