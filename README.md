# Code for paper "Integrated Speech and Gesture Synthesis"(ICMI 2021)
paper link: https://arxiv.org/pdf/2108.11436.pdf

## Scope
This repository reproduces the Tacotron2-ISG models from the paper. 

## Go step-by-step OR use this repository flexibly
The steps outlined below should reproduce the results from the paper. 
But one can also select parts (steps) that are needed for their own purpose. 
For example, if you want to try your own model but on the same data and data preprocessing
as us, then just follow the corresponding data preparation steps. Or if you have your
own dataset, you can start at Step 2-3 to train our models on your dataset.

## Model checkpoints not provided
We do not provide checkpoints because the models are trained on a licensed dataset.

## Step 1: Get the data
We use [Trinity Speech-Gesture Dataset](https://trinityspeechgesture.scss.tcd.ie/) (part I). 
The usage of the dataset must be licensed through their official website.
Please obtain the proper license if you wish to use the dataset and reproduce the results from 
our paper. The dataset will be referred as "TSG dataset" or the dataset from now on. However, 
the training pipeline works with any dataset with a proper format as specified in Step 2-3.

After obtaining a license and gaining access to the dataset repository, there should be two folders indicating PART I of 
the dataset and PART II of the dataset. We use PART I. Inside the PART I folder, there should be a folder called 
GENEA_2020. This is a processed version of the PART I data which contains manually corrected transcripts, aligned speech
and motion, and already split train/test sets. This is the data that is used here.

IMPORTANT: If you use TSG dataset FOR ANY REASON, with or without our repo, please make appropriate citations as 
indicated in the official TSG dataset repository. If you use GENEA_2020 data, please make appropriate citations of
BOTH TSG dataset and GENEA 2020 challenge. All up-to-date citation information is contained in the TSG dataset 
repository.

## Step 2: Segment the speech-gesture data
The TSG dataset contains 10-min episodes of recorded speech-gesture. They need to be
cut into segments of less than 12 seconds. This is because Tacotron-2 (the TTS module of
our models) is most commonly trained with speech segments not longer than 12 seconds. Training with longer segments
may result in breaking attention etc.

We segmented speech into <12s segments with a method called "breathgroup bi-grams" in the paper, but 
unfortunately this part of the code is not ready to be released. Here, we provide a simple segmentation heuristic of
random windowing, that is to segment using a sliding window with random size and random step size (both sampled 
uniformly within pre-determined min and max at every step).

Change the path variables at the top of the file seg.py and run the script ```python seg.py```, this will:
- Segment both train and test speech episodes into <12s segments by random sliding windowing
- create train/val/test filelists that will be used in training

## Step 3: Preprocess the motion data
The motion capture data is in bvh format, which represents motion by specifying joint rotation in euler angles in a
pre-defined skeleton. Euler angles are not fit for machine learning due to its ambiguity that is the same rotation can
be represented by different Euler angles. Thus, as many other works in motion synthesis, we extract exponential map 
representation from original euler angles. On top of this, we normalize the obtained exponential map representation per
feature dimension, in which the mean and variance are obtained from train set.

Change the path variables at the top of the file get_gesture_exp.py:
- [BVH_DIR] should be the same as [OUT_BVH_DIR] specified in seg.py. 
- [FLISTS] should be the same train/dev/test filelist paths as specified in seg.py.

Run get_gesture_exp.py ```python get_gesture_exp.py``` after path variables have been correctly specified. 
This will first extract exponential map representation from each bvh file and then normalize the resulting data, 
where mean and variance are obtained only from motion segments in train set.

NOTE: Running get_gesture_exp.py can take a long time.

## Step 4: Speech-only training
Download LJSpeech-trained Tacotron 2 model from https://github.com/NVIDIA/tacotron2.

Change following arguments in model_configs/speech_only.yaml:
- [training_files] should be [TRAIN_OUT_FILELIST] specified in seg.py or your own train filelist.
- [validation_files] should be [DEV_OUT_FILELIST] specified in seg.py or your own val filelist.
- [wav_directory] should be [WAV_DIR] specified in seg.py or where the wavs is store for your own dataset.
- [gesture_dir] should be [BVH_DIR] specified in seg.py or where normalized motion files are store for you own dataset. 
- [tt2_checkpoint] should be path to the downloaded LJSpeech-trained Tacotron 2 model checkpoint.
- [batch_size] should be adjusted to maximize GPU usage.

After the above changes have been applied, train speech-only model by running
```
python train.py --model_config model_configs/speech_only.yaml
```

For multi-gpu training, modify model_configs/speech_only_dt.yaml similarly as model_configs/speech_only.yaml and specify
- [n_gpus] as the number of gpus to use 
- set [distributed] as True

Then run,
```
python -m multiproc train.py --model_config model_configs/speech_only.yaml
```

Convergence can be checked in several ways:
- Loss curve flattens and hovers around 0.20
- ~ 150,000 iterations at batch size 20
- Synthesized speech sounds good

Loss curve is plotted by tensorboard. Run tensorboard:
```tensorboard --logdir .```

Speech-only synthesis:
```
python test_isg.py --speech_only --not_use_half --model_config model_configs/speech_only.yaml --test_config test_configs/test_latest.yaml --num_synth 10
```

test_isg.py can also take a text file as input, in which each line is taken as an input sentence.
We provide, as part of this repo, evaluation input sentences generated by a finetuned GPT2 model used in evaluations in
the paper. Please see paper Section 4.2 for more details. To use these as input sentences, run
```
python test_isg.py --speech_only --not_use_half --model_config model_configs/speech_only.yaml --test_config test_configs/test_latest.yaml --text_inputs data/gpt2_generated_prompts.txt
```

IMPORTANT: The GPT2 generated sentences are for testing and evaluation of speech-only and ISG models only. They do not 
represent our view in any way.

## Step 5: ISG training
The paper introduces two ways to train proposed Tacotron2-ISG model, namely ST-Tacotron2-ISG and CT-Tacotron2-ISG.
The model configs are provided in model_configs/ct_isg.yaml and model_configs/st_isg.yaml.
To train the models, simply change path variables and run train.py with changed yaml file similarly to Step 4.

Here, the process is shown for training ST-Tacotron-ISG. 

Change following path variables in model_configs/isg_st.yaml (first 4 are the same as in Step 4):
- [training_files] should be [TRAIN_OUT_FILELIST] specified in seg.py or your own train filelist.
- [validation_files] should be [DEV_OUT_FILELIST] specified in seg.py or your own val filelist.
- [wav_directory] should be [WAV_DIR] specified in seg.py or where the wavs is store for your own dataset.
- [gesture_dir] should be [BVH_DIR] specified in seg.py or where normalized motion files are store for you own dataset. 
- [tt2_checkpoint] should be path to the trained speech-only model from Step 4. Ideally this should be 
- [batch_size] should be adjusted to maximize GPU usage. Note that ISG training takes more gpu memory per sample due to
added gesture dimensions, so batch size should generally be smaller than specified in Step 4 if using same gpu(s).
  
After the above changes have been applied, run
```
python train.py --model_config model_configs/isg_st.yaml
```

Multi-gpu training is similar to Step 4.

Convergence can be checked in several ways:
- Loss curve flattens and hovers around 0.24 (for both ct and st models)
- Synthesized speech-gesture sounds and looks good

ISG synthesis:

Use sentences from dev set in the original dataset
```
python test_isg.py --not_use_half --model_config model_configs/isg_st.yaml --test_config test_configs/test_latest.yaml --num_synth 10
```

Use GPT2 generated sentences
```
python test_isg.py --not_use_half --model_config model_configs/isg_st.yaml --test_config test_configs/test_latest.yaml --text_inputs data/gpt2_generated_prompts.txt
```

## License
See LICENSE file.
