import sys
sys.path.append('waveglow/')
# import numpy as np
# import torch
# import csv
# import os
from g2p_en import G2p
# import re
# import itertools as it
# import shutil

from test_utils import *
# from synth_df_insert import text_to_tensor
# import soundfile
import librosa
# from text import text_to_sequence
# sys.path.append('process_motion')
from get_gesture_exp import std_exp_to_bvh
import pdb

def text_to_tensor(text, text_cleaners, g2p):
    text = get_phon_seq(text, g2p)
    text_norm = torch.IntTensor(text_to_sequence(text, text_cleaners))
    return text_norm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str,
                        required=True, help='yaml containing model config')
    parser.add_argument('--test_config', type=str,
                        required=True, help='yaml containing test config')
    parser.add_argument('--render_list', type=str,
                        default=None, help='text file containing list of segments to test.')
    parser.add_argument('--text_inputs', type=str,
                        default=None, help='text file containing text input to cosg.')
    parser.add_argument('--text_inputs_fname_prefix', type=str,
                        default='gp', help='prefix to output fname for text inputs.')
    parser.add_argument('--not_use_half', action='store_true'
                        , help='not use half precision at waveglow')
    parser.add_argument('--count_parameters', action='store_true',
                        help='count number of parameters and return.')
    parser.add_argument('--synth_dir', type=str, default='./synth')
    parser.add_argument('--num_synth', type=int, default=-1, help='-1 if synth all inputs')
    parser.add_argument('--waveglow_path', type=str, default='models/waveglow_256channels_universal_v5.pt')

    parser.add_argument('--gesture_out_fps', type=float,
                        default=20.0, help='Gesture lstm subsample rate.')
    parser.add_argument('--gesture_smoothing', type=str,
                        default=None, help='smoothing methods: box, normal')
    parser.add_argument('--gesture_smoothing_half_ws', type=int,
                        default=5, help='smoothing methods: box, normal')

    parser.add_argument('--scaler_fpath', type=str, default='data/bvh/std_exp_scaler.sav')
    parser.add_argument('--data_pipe_fpath', type=str, default='data/bvh/data_pipe-train.sav')

    parser.add_argument('--speech_only', action='store_true')

    args = parser.parse_args()
    model_config = yaml.load(open(args.model_config, 'r'))
    test_config = yaml.load(open(args.test_config, 'r'))
    model_name = os.path.basename(args.model_config)
    model_name = os.path.splitext(model_name)[0]
    # pdb.set_trace()

    if not os.path.isdir(args.synth_dir):
        os.mkdir(args.synth_dir)
    out_dir = f"{args.synth_dir}/{model_name}/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    hparams = config_hparams(model_config, test_config)
    # pdb.set_trace()
    model, load_iter, max_iter = load_test_model(model_name, test_config["iter"], hparams, return_iter=True)
    model_name = model_name  + f"_i{load_iter}m{max_iter}"
    if args.count_parameters:
        print("Total number of trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        exit(0)

    # prepare data loaders
    # re-init cpu side seed to insure shuffle seed in dataloader is same from different models
    torch.manual_seed(hparams.seed)
    # pdb.set_trace()
    # pdb.set_trace()
    # testing through the data set
    # pdb.set_trace()
    wav_lens = []
    len_diffs = []
    render_list = None
    num_synth = None
    if args.render_list is not None:
        # pdb.set_trace()
        render_list = []
        with open(args.render_list, "r") as f:
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]
                render_list.append(line)
        num_synth = len(render_list)
    text_inputs = None
    if args.text_inputs is not None:
        # pdb.set_trace()
        text_inputs = []
        with open(args.text_inputs, "r") as f:
            for line in f:
                if line[-1] == '\n':
                    line = line[:-1]
                text_inputs.append(line)
        num_synth = len(text_inputs)
        g2p = G2p()
    else:
        train_loader, val_loader, collate_fn = prepare_dataloaders(hparams, True)
        num_synth = len(val_loader)
        val_iterator = val_loader._get_iterator()

    mel_outputs = []
    out_wav_paths = []
    out_mel_paths = []
    synth_timer = SynthTimer()
    for i in range(num_synth):
        if test_config["num_samples"] >= 0 and i == test_config["num_samples"]:
            break

        if not text_inputs is None:
            if i >= len(text_inputs):
                break

            input_text = text_to_tensor(text_inputs[i], hparams.text_cleaners, g2p)
            sample_name = f"{args.text_inputs_fname_prefix}{i}"
        else:
            pdb.set_trace()
            batch = val_iterator.__next__()
            audio_paths = batch[-2]
            audio_path = audio_paths[0]
            sample_name = os.path.splitext(os.path.basename(audio_path))[0]
            if render_list is not None and sample_name not in render_list:
                continue

        out_wav_name = "{}-{}".format(model_name, sample_name)
        out_mel_name = "{}-{}-mel".format(model_name, sample_name)
        out_gesture_name = "{}-{}-gesture".format(model_name, sample_name)
        out_wav_path = out_dir + out_wav_name + '.wav'
        out_mel_path = out_dir + out_mel_name + '.npy'
        out_gesture_path = out_dir + out_gesture_name + '.bvh'
        out_wav_paths.append(out_wav_path)
        print(i, sample_name)

        # pdb.set_trace()
        if text_inputs is None:
            x, y = model.parse_batch_for_eval(batch)
            # pdb.set_trace()
            synth_timer.start()
            y_pred = model.inference(x[0], speaker_feat=None) #, feats=x[-2], step_feats=x[-1])
        else:
            synth_timer.start()
            y_pred = model.inference(input_text.long().cuda().unsqueeze(0), speaker_feat=None)
        synth_timer.end()

        if "time_only" in test_config.keys() and test_config["time_only"]:
            continue

        # pdb.set_trace()
        mel_output, gesture_out = y_pred[1][:,:hparams.n_mel_channels].detach().cpu().numpy(), \
                                  y_pred[1][:,hparams.n_mel_channels:].detach().cpu().numpy()
        if not args.speech_only:
            std_exp_to_bvh(gesture_out, out_gesture_path, out_fps=args.gesture_out_fps, scaler_fpath=args.scaler_fpath,
                           data_pipe_fpath=args.data_pipe_fpath)
        mel_outputs.append(mel_output)
        # del x, y, y_pred
        # break
    synth_timer.output_json(out_dir + "synth_times.json")
    if "time_only" in test_config.keys() and test_config["time_only"]:
        exit(0)

    del model
    torch.cuda.empty_cache()
    # pdb.set_trace()

    waveglow, denoiser = init_wavglow(args.waveglow_path, use_half=(not args.not_use_half))
    set_sigma = 0.7
    set_denoise = 0.06
    for mel_output, out_wav_path in zip(mel_outputs, out_wav_paths):
        if args.not_use_half:
            mel_output = torch.tensor(mel_output).cuda()
        else:
            mel_output = torch.tensor(mel_output).cuda().half()
        audio_denoised = denoiser(waveglow.infer(mel_output, sigma=set_sigma), strength=set_denoise)[:, 0]

        # NOTE: either of the following wav saving could work depending on which os is running
        librosa.output.write_wav(out_wav_path, audio_denoised.cpu().numpy().T, hparams.sampling_rate)
        # soundfile.write(out_wav_path, audio_denoised.cpu().numpy().T, hparams.sampling_rate)