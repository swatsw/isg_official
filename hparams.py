import os
import json
import datetime

from text import symbols

class JsonConfig(dict):
    """
    Adapted from https://github.com/simonalexanderson/StyleGestures: config.py

    The configures will be loaded and dumped as json file.
    The Structure will be maintained as json.
    [TODO]: Some `asserts` can be make by key `__assert__`
    """
    Indent = 2

    def __init__(self, *argv, **kwargs):
        super().__init__()
        super().__setitem__("__name", "default")
        # check input
        assert len(argv) == 0 or len(kwargs) == 0, (
            "[JsonConfig]: Cannot initialize with"
            " position parameters (json file or a dict)"
            " and named parameters (key and values) at the same time.")
        if len(argv) > 0:
            # init from a json or dict
            assert len(argv) == 1, (
                "[JsonConfig]: Need one positional parameters, found two.")
            arg = argv[0]
        else:
            arg = kwargs
        # begin initialization
        if isinstance(arg, str):
            super().__setitem__("__name",
                                os.path.splitext(os.path.basename(arg))[0])
            with open(arg, "r") as load_f:
                arg = json.load(load_f)
        if isinstance(arg, dict):
            # case 1: init from dict
            for key in arg:
                value = arg[key]
                if isinstance(value, dict):
                    value = JsonConfig(value)
                super().__setitem__(key, value)
        else:
            raise TypeError(("[JsonConfig]: Do not support given input"
                             " with type {}").format(type(arg)))

    def __setattr__(self, attr, value):
        raise Exception("[JsonConfig]: Can't set constant key {}".format(attr))

    def __setitem__(self, item, value):
        raise Exception("[JsonConfig]: Can't set constant key {}".format(item))

    def __getattr__(self, attr):
        return super().__getitem__(attr)

    def __str__(self):
        return self.__to_string("", 0)

    def __to_string(self, name, intent):
        ret = " " * intent + name + " {\n"
        for key in self:
            if key.find("__") == 0:
                continue
            value = self[key]
            line = " " * intent
            if isinstance(value, JsonConfig):
                line += value.__to_string(key, intent + JsonConfig.Indent)
            else:
                line += " " * JsonConfig.Indent + key + ": " + str(value)
            ret += line + "\n"
        ret += " " * intent + "}"
        return ret

    def __add__(self, b):
        assert isinstance(b, JsonConfig)
        for k in b:
            v = b[k]
            if k in self:
                if isinstance(v, JsonConfig):
                    super().__setitem__(k, self[k] + v)
                else:
                    if k == "__name":
                        super().__setitem__(k, self[k] + "&" + v)
                    else:
                        assert v == self[k], (
                            "[JsonConfig]: Two config conflicts at"
                            "`{}`, {} != {}".format(k, self[k], v))
            else:
                # new key, directly add
                super().__setitem__(k, v)
        return self

    # def update_by_dict(self, b):
    #     assert isinstance(b, dict)
    #     for k,v in b.items():
    #         assert not isinstance(v, dict)
    #         super().__setitem__(k, v)

    def date_name(self):
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        return date + "_" + super().__getitem__("__name") + ".json"

    def dump(self, dir_path, json_name=None):
        if json_name is None:
            json_name = self.date_name()
        json_path = os.path.join(dir_path, json_name)
        with open(json_path, "w") as fout:
            print(str(self))
            json.dump(self.to_dict(), fout, indent=JsonConfig.Indent)

class HParams(JsonConfig):
    def __init__(self, *argv, **kwargs):
        # self.has_init = False
        super().__init__(*argv, **kwargs)
        # self.has_init = True

    def __setattr__(self, attr, value):
        # if self.has_init:
        assert attr in self.keys()
        foo = {attr:value}
        self.update(foo)

    def __setitem__(self, item, value):
        # if self.has_init:
        assert item in self.keys()
        foo = {item: value}
        self.update(foo)

# def dict_from_string(s):
#     assert isinstance(s, str)
#     s = [x.split("=") for x in s.split(",")]
#     d = {}
#     for kv in s:
#         d[kv[0]] =

def create_hparams(*args):
    """Create model hyperparameters.
    Args must be dicts.
    The dicts with update hparam dict in order."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=50000,
        iters_per_checkpoint=2500,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        n_gpus=1,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=[''],
        shuffle_dataloader=True,
        freeze_encoder=False,

        native_fp16=False,
        warm_start_tt2=True,
        tt2_checkpoint="models/tacotron2_statedict.pt",

        saved_checkpoint=None, # use this when restarting ISG training

        output_directory="output",
        log_directory="logdir",
        wav_directory=".",

        ################################
        # Additional Factor Parameters #
        ################################
        # Speaker embedding
        speakers=False,
        n_speakers=2,
        speaker_embedding_dim=8,
        
        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False, # if True, calculated mel is saved during first iter of training and loaded later
        training_files='filelists/Liam_v4_feat_train_filelist.txt',
        validation_files='filelists/Liam_v4_feat_val_filelist.txt',
        #training_files='filelists/duo_train_filelist.txt',
        #validation_files='filelists/duo_val_filelist.txt',
        text_cleaners=['english_cleaners'],
        return_audio_path=False, # return audio path as part of data loader function

        gesture_dir="data/gesture/",
        gesture_file_suffix="_std_exp",
        gesture_fps=60,
        # gesture files are loaded as f"{hparams.gesture_dir}/{audioname}{hparams.gesture_file_suffix}.npy"

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,


        # gesture output branch
        gesture_out=False,
        gesture_dim=0,
        gesture_to_prenet=False,

        # gesture has a separate lstm
        gesture_lstm=False,
        gesture_lstm_dim=1024,
        p_gesture_lstm_dropout=0.1,
        gesture_prenet_dim=256,
        p_gesture_prenet_dropout=0.5, # this default is the same as speech prenet dropout
        gesture_prenet_no_dropout_at_test=True, # speech prenet dropout at both train and test
        gesture_use_postnet=False,
        gesture_lstm_use_prenet=False,
        encoder_embedding_to_gesture_lstm=True,
        gesture_lstm_subsample=1, # number steps of attention rnn for each gesture lstm step
        gesture_use_moglow=False,
        gesture_moglow_json_config="",

        gesture_lstm_l=1,
        attn_to_gesture_use_mlp=False,
        attn_to_gesture_mlp_l=1,
        attn_to_gesture_mlp_dim=512,
        p_dropout_attn_to_gesture_mlp=0.0,

        # gesture scheduled sampling
        gesture_use_scheduled_sampling=False,
        gesture_scheduled_sampling_epochs=[1e5, 0], # [full teacher forcing, linear decay(full autoregressive afterwards)]
        gesture_scheduled_sampling_min=0.0,  # scheduled sampling p drop to and stay at this value
        gesture_scheduled_sampling_max=1.0,

        freeze_tt2=False,

        # speech-gesture gan
        SG_gan=False,
        separate_d_optimizer=False,
        add_misalign=False,
        misalign_min=10,
        misalign_max=40,
        p_SG_d_dropout=0.1,
        SG_gan_loss_w=1.0, # gan loss weight
        SG_d_create_misalign=False,
        SG_d_misalign_min_frame=20,
        SG_d_misalign_max_frame=80,

        lstm_d_use_both_hc=True,
        lstm_d_dim=512,
        lstm_d_l=1,

        # gesture lstm at mel output
        glstm_at_mel=False,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-5,
        d_learning_rate=1e-5, # discriminator lr in SG_gan=True
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        clip_grad=True,
        batch_size=20,
        mask_padding=True  # set model's padded outputs to padded values
    )

    for d in args:
        assert isinstance(d, dict)
        hparams.update(d)

    return hparams
