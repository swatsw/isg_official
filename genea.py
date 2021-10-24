from scipy.io.wavfile import read, write
from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *

import os
import json
import re
from g2p_en import G2p
import librosa
import numpy as np
import soundfile

FRONT_APPEND_SILENCE = 0.2 # in seconds
MAX_SILENCE_RATIO = 0.2 # at most this much silence in an utterance
PRIVACY_TOKEN = "Token"

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

class GENEA():
    def __init__(self, wav_folder, bvh_folder, text_folder, wav_out_sr=22050):
        wav_files = os.listdir(wav_folder)
        bvh_files = os.listdir(bvh_folder)
        text_files = os.listdir(text_folder)

        wav_episodes = [os.path.splitext(x)[0] for x in wav_files]
        bvh_episodes = [os.path.splitext(x)[0] for x in bvh_files]
        text_episodes = [os.path.splitext(x)[0] for x in text_files]
        episodes = list(set(wav_episodes + bvh_episodes + text_episodes))
        for x in episodes:
            assert (x in bvh_episodes) and \
                   (x in bvh_episodes) and \
                   (x in text_episodes), f"{x} does not have either wav, bvh, or text file"

        self.episodes = episodes
        # self.episodes = episodes[:1] # TODO: remove after debug
        # self.episodes = ["Recording_021"] # TODO: remove after debug

        self.wavs = {}
        self.wav_durs = {}
        self.bvhs = {}
        self.texts = {}
        self.text_seg_times = {}
        self.curr_word_i = {}
        bvh_parser = BVHParser()
        sr_all = None
        bvh_framerate_all =None
        for episode in self.episodes:
            print(f"loading {episode} ...")

            wav_fpath = f"{wav_folder}/{episode}.wav"
            wav, sr = soundfile.read(wav_fpath)
            if sr_all is None:
                sr_all = sr
            else:
                assert sr_all == sr, f"not uniform sampling rate: {wav_fpath}, {sr_all} {sr}"
                sr_all = sr
            self.wavs[episode] = wav
            self.wav_durs[episode] = len(wav) / sr

            bvh_fpath = f"{bvh_folder}/{episode}.bvh"
            bvh = bvh_parser.parse(bvh_fpath)
            if bvh_framerate_all is None:
                bvh_framerate_all = bvh.framerate
            else:
                assert abs(bvh_framerate_all-bvh.framerate) < 1e-3, \
                    f"not uniform frame rate: {bvh_fpath}, {bvh_framerate_all} {bvh.framerate}"
            self.bvhs[episode] = bvh

            text_fpath = f"{text_folder}/{episode}.json"
            text = json.load(open(text_fpath, "r"))
            text_contiguous = []
            for text_seg in text:
                assert len(text_seg['alternatives']) == 1
                words = text_seg['alternatives'][0]['words']
                for w in words:
                    w['start_time'] = float(w['start_time'][:-1])
                    w['end_time'] = float(w['end_time'][:-1])
                text_contiguous.extend(words)
            self.check_text_time(text_contiguous, episode)
            self.texts[episode] = text_contiguous
        self.refresh_curr_word_i()

        self.orig_sr = sr_all
        self.bvh_framerate = bvh_framerate_all

        self.wav_folder = wav_folder
        self.bvh_folder = bvh_folder
        self.bvh_writer = BVHWriter()

        self.g2p = G2p()
        self.wav_out_sr = wav_out_sr

    def check_text_time(self, text, episode):
        prior_endtime = None
        for i, t in enumerate(text):
            if prior_endtime is None:
                prior_endtime = t['end_time']
            else:
                if not t['start_time'] >= prior_endtime:
                    print(f"{episode} {i}, {text[i]}, {text[i-1]}")
                prior_endtime = t['end_time']

    def refresh_curr_word_i(self):
        self.curr_word_i = {x:0 for x in self.episodes}

    def get_wav_dur(self, episode):
        return self.wav_durs[episode]

    def segment_transcript(self, episode, starttime, dur, max_silence_ratio=MAX_SILENCE_RATIO):
        '''
        assumes transcript segments are in order of time
        :return:
        '''
        word_i = self.curr_word_i[episode]
        text = self.texts[episode]
        silence_dur = 0.0
        total_dur = 0.0
        true_starttime = None
        utter_text = []
        while true_starttime is None:
            if not word_i < len(text):
                return (starttime, None, None)

            if starttime < text[word_i]['end_time']:
                # assert text[word_i]['end_time'] - text[word_i]['start_time'] < dur,\
                #     f"{text[word_i]['end_time']}, {text[word_i]['start_time']}, {text[word_i]}"
                true_starttime = text[word_i]['start_time']
                self.curr_word_i[episode] = word_i + 1 # update starting word_i in episode
                break
            word_i += 1

        while word_i < len(text):
            # word_starttime = text[word_i]['start_time']
            word_endtime = text[word_i]['end_time']
            # word_dur = word_endtime - word_starttime

            if word_endtime - true_starttime > dur: # utterance duration exceeded
                break
            else:
                total_dur = word_endtime - true_starttime
                utter_text.append(text[word_i]['word'])
                word_i += 1

                if word_i < len(text):  # shoudl already incremented to next word_i
                    next_word_starttime = text[word_i]['start_time']
                    silence_dur += next_word_starttime - word_endtime # silence between curr and next word
                    total_dur = next_word_starttime - true_starttime

        if true_starttime is None:
            true_starttime = starttime

        if total_dur == 0 or total_dur < dur - 2:
            return (true_starttime, None, None)

        if silence_dur / total_dur > max_silence_ratio or PRIVACY_TOKEN in utter_text:
            return (true_starttime, None, None)
        else:
            return (true_starttime, total_dur, " ".join(utter_text))

    def segment(self, episode, starttime, dur, segment_name=None, wav_out_dir=None, bvh_out_dir=None):
        true_starttime, true_dur, utter_text = self.segment_transcript(episode, starttime, dur)
        if utter_text is None:
            return (true_starttime, utter_text)
        true_endtime = true_starttime + true_dur
        print(f"{episode}: {segment_name}, word_i: {self.curr_word_i[episode]}, "
              f"{true_starttime} -> {true_endtime}")
        utter_text = get_phon_seq(utter_text, self.g2p)

        wav = self.wavs[episode]
        start_frame = int(true_starttime * self.orig_sr)
        end_frame = int((true_endtime) * self.orig_sr)
        if wav_out_dir is not None and segment_name is not None:
            out_wav = np.array(wav[start_frame:end_frame])
            out_wav = librosa.resample(out_wav, target_sr=self.wav_out_sr, orig_sr=self.orig_sr)
            soundfile.write(f"{wav_out_dir}/{segment_name}.wav", out_wav, self.wav_out_sr)

        new_bvh = self.bvhs[episode].clone()
        bvh_start_frame = int(true_starttime / self.bvh_framerate)
        bvh_end_frame = int((true_endtime) / self.bvh_framerate)
        new_bvh.values = new_bvh.values.iloc[bvh_start_frame:bvh_end_frame]
        if bvh_out_dir is not None and segment_name is not None:
            self.bvh_writer.write(new_bvh, open(f"{bvh_out_dir}/{segment_name}.bvh", "w"))

        return (true_starttime, utter_text)
