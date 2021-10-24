import random
import numpy as np
import torch
import torch.utils.data
import ast

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence

import pdb
import csv
import os
import math
import pickle as pkl

def align_gesture_with_mel(gesture_arr, hop_length=256, sampling_rate=22050, gesture_fps = 60):
    audio_frame_hop_time = hop_length / sampling_rate
    gesture_frame_time = 1.0 / gesture_fps
    total_gesture_time = gesture_arr.shape[0] * gesture_frame_time
    num_out_frame = math.floor(total_gesture_time / audio_frame_hop_time)
    align_indices = np.arange(num_out_frame, dtype=np.float32)
    align_indices *= audio_frame_hop_time
    align_indices /= gesture_frame_time
    align_indices = np.rint(align_indices).astype(np.int)
    out_arr = gesture_arr[align_indices, :]
    return out_arr

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
        
    """
    def __init__(self, audiopaths_and_text, hparams, feats=None, hand_feats_paths=None):
        self.wav_dir = hparams.wav_directory
        self.audiopaths_and_text_fname = audiopaths_and_text
        self.audiopaths_and_text = load_filepaths_and_text(self.audiopaths_and_text_fname)
        # self.audiopaths_and_text = self.audiopaths_and_text[:500] # TODO: REMOVE AFTER DEBUG
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        if hparams.speakers:
            self.feats = feats
            self.speakers = True
        else:
            self.speakers= False
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

        self.gesture_out = hparams.gesture_out
        self.gesture_dim = hparams.gesture_dim
        self.gesture_dat = None
        if self.gesture_out:
            self.gesture_dat = []
            for audiopath in self.audiopaths_and_text:
                audioname = audiopath[0]
                audioname = os.path.basename(audioname)
                audioname = os.path.splitext(audioname)[0]

                gesture_path = f"{hparams.gesture_dir}/{audioname}{hparams.gesture_file_suffix}.npy"
                gesture_arr = np.load(gesture_path)

                aligned_gesture = align_gesture_with_mel(gesture_arr, hparams.hop_length, hparams.sampling_rate,
                                                         hparams.gesture_fps)
                mel = self.get_mel(audiopath[0])
                mel_len = mel.T.shape[0]
                if aligned_gesture.shape[0] >= mel_len: # gesture is cut if longer than mel
                    aligned_gesture = aligned_gesture[:mel_len]
                elif aligned_gesture.shape[0] < mel_len: # gesture is padded is shorter than mel
                    padded_gesture = np.zeros((mel_len, aligned_gesture.shape[1]))
                    padded_gesture[:aligned_gesture.shape[0]] = aligned_gesture
                    padded_gesture[aligned_gesture.shape[0]:] = aligned_gesture[-1]
                    aligned_gesture = padded_gesture
                self.gesture_dat.append(torch.tensor(aligned_gesture))

    def get_mel_text_pair(self, audiopath_and_text, index):
        # separate filename and text
        speaker_feats = None
        if self.speakers:
            audiopath, text, speaker_feats = audiopath_and_text[0], audiopath_and_text[1], int(audiopath_and_text[2])
        else:
            audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)

        gesture_arr = None if not self.gesture_out else self.gesture_dat[index]
        return (text, mel, speaker_feats, audiopath, gesture_arr)
    text_i_in_batch = 0
    mel_i_in_batch = 1
    speaker_feats_i_in_batch = 2
    audiopath_i_in_batch = 3
    gesture_arr_i_in_batch = 4

    def get_mel(self, filename):
        if self.wav_dir != "":
            filename = f"{self.wav_dir}/{filename}"
        no_ext, ext = os.path.splitext(filename)
        assert ext == ".wav", "Only supports wav files."
        mel_fname = f"{no_ext}-sr{self.stft.sampling_rate}.npy"
        if (not os.path.isfile(mel_fname)) or (not self.load_mel_from_disk):
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            self.last_audio_length = audio_norm.shape[1] / sampling_rate
            if self.load_mel_from_disk:
                np.save(mel_fname, melspec, allow_pickle=False)
                print(f"saved {mel_fname}")
        else:
            # pdb.set_trace()
            melspec = torch.from_numpy(np.load(mel_fname))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index], index)

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, speakers=False, gesture_out=False, return_audio_path=False):
        self.n_frames_per_step = n_frames_per_step
        self.speakers=speakers
        self.gesture_out=gesture_out
        self.return_audio_path = return_audio_path

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, feats(text-level), step_feats(mel-frame-level)]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[TextMelLoader.text_i_in_batch]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][TextMelLoader.text_i_in_batch]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][TextMelLoader.mel_i_in_batch].size(0)
        max_target_len = max([x[TextMelLoader.mel_i_in_batch].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_feats = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][TextMelLoader.mel_i_in_batch]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            if self.speakers:
                speaker_feats[i] = batch[ids_sorted_decreasing[i]][TextMelLoader.speaker_feats_i_in_batch]

        # collate and pad gesture arr
        gesture_arr_padded = None
        if self.gesture_out:
            gesture_arr_padded = \
                torch.FloatTensor(len(batch), batch[0][TextMelLoader.gesture_arr_i_in_batch].shape[-1], max_target_len)
            gesture_arr_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                gesture_arr = batch[ids_sorted_decreasing[i]][TextMelLoader.gesture_arr_i_in_batch].T
                gesture_arr_padded[i,:,:gesture_arr.shape[1]] = gesture_arr

        audio_paths=None
        if self.return_audio_path:
            audio_paths = [x[TextMelLoader.audiopath_i_in_batch] for x in batch]
        return text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, speaker_feats, audio_paths, gesture_arr_padded

    audio_paths_i_in_collated_batch = -2