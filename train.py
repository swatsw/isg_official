import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams

import pdb
import yaml

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # pdb.set_trace()
    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams, val_loader_only=False):
    # Get data, data loaders and collate function ready
    shuffle = False
    if not val_loader_only:
        trainset = TextMelLoader(hparams.training_files, hparams)
        if hparams.distributed_run:
            train_sampler = DistributedSampler(trainset)
            shuffle = False
        else:
            train_sampler = None
            shuffle = hparams.shuffle_dataloader

    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step, hparams.speakers, hparams.gesture_out, hparams.return_audio_path)

    if val_loader_only:#only at test time
        val_loader = DataLoader(valset, num_workers=0, shuffle=shuffle,
                                sampler=None,
                                batch_size=hparams.batch_size, pin_memory=False,
                                drop_last=True, collate_fn=collate_fn)
        return None, val_loader, collate_fn
    else:
        train_loader = DataLoader(trainset, num_workers=0, shuffle=shuffle,
                                  sampler=train_sampler,
                                  batch_size=hparams.batch_size, pin_memory=False,
                                  drop_last=True, collate_fn=collate_fn)
        return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run or hparams.native_fp16:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    add_encoder_dim = model.decoder.encoder_embedding_dim \
                      - model_dict['decoder.attention_layer.memory_layer.linear_layer.weight'].shape[1]
    add_attention_rnn_in_dim = model.decoder.attention_rnn.input_size - model_dict['decoder.attention_rnn.weight_ih'].shape[1]
    add_decoder_rnn_in_dim = model.decoder.decoder_rnn.input_size - model_dict['decoder.decoder_rnn.weight_ih'].shape[1]
    add_linear_proj_dim = model.decoder.linear_projection.linear_layer.in_features \
                          - model_dict['decoder.linear_projection.linear_layer.weight'].shape[1]
    add_prenet_dim = model.decoder.prenet.layers[0].linear_layer.in_features \
                    - model_dict['decoder.prenet.layers.0.linear_layer.weight'].shape[1]
    # if adding a new feature to the character embedding, add zero weight vectors to loaded model
    if add_attention_rnn_in_dim > 0:
        with torch.no_grad():
            model_dict['decoder.attention_rnn.weight_ih'] = \
                torch.cat((model_dict['decoder.attention_rnn.weight_ih'],
                torch.zeros(model_dict['decoder.attention_rnn.weight_ih'].size()[0], add_attention_rnn_in_dim)), 1)
    if add_decoder_rnn_in_dim > 0:
        with torch.no_grad():
            model_dict['decoder.decoder_rnn.weight_ih'] = \
                torch.cat((model_dict['decoder.decoder_rnn.weight_ih'],
                torch.zeros(model_dict['decoder.decoder_rnn.weight_ih'].size()[0], add_decoder_rnn_in_dim)), 1)
    if add_encoder_dim > 0:
        with torch.no_grad():
            model_dict['decoder.attention_layer.memory_layer.linear_layer.weight'] = \
                torch.cat((model_dict['decoder.attention_layer.memory_layer.linear_layer.weight'], 
                torch.zeros(model_dict['decoder.attention_layer.memory_layer.linear_layer.weight'].size()[0], add_encoder_dim)), 1)
            model_dict['decoder.gate_layer.linear_layer.weight'] = \
                torch.cat((model_dict['decoder.gate_layer.linear_layer.weight'], 
                torch.zeros(model_dict['decoder.gate_layer.linear_layer.weight'].size()[0], add_encoder_dim)), 1)
    if add_linear_proj_dim > 0:
        with torch.no_grad():
            model_dict['decoder.linear_projection.linear_layer.weight'] = \
                torch.cat((model_dict['decoder.linear_projection.linear_layer.weight'],
                torch.zeros(model_dict['decoder.linear_projection.linear_layer.weight'].size()[0],
                                   add_linear_proj_dim)), 1)
    if add_prenet_dim > 0:
        with torch.no_grad():
            model_dict['decoder.prenet.layers.0.linear_layer.weight'] = \
                torch.cat((model_dict['decoder.prenet.layers.0.linear_layer.weight'],
                           torch.zeros(model_dict['decoder.prenet.layers.0.linear_layer.weight'].size()[0],
                                       add_prenet_dim)), 1)

    if not ignore_layers[0] == '':
        ignore_layers = [x[1:] for x in ignore_layers if x[0] == "'"]
        ignore_layers = [x[:-1] for x in ignore_layers if x[-1] == "'"]
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}

    curr_model_dict = model.state_dict()
    curr_model_dict.update(model_dict)
    has_new_weights = (not len(curr_model_dict.keys()) == len(model_dict.keys()))
    model_dict = curr_model_dict
    model.load_state_dict(model_dict, strict=True)
    return model, (add_attention_rnn_in_dim > 0) or \
        (add_decoder_rnn_in_dim > 0) or \
        (add_encoder_dim > 0) or \
        (add_linear_proj_dim > 0) or \
        (add_prenet_dim > 0) or \
        has_new_weights


def load_checkpoint(checkpoint_path, model, optimizer, d_optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    # ckpt_weight_names = list(checkpoint_dict['state_dict'].keys())
    # if 'decoder.gesture_lstm_rnn.weight_ih' in ckpt_weight_names:
    #     for weight_name in ckpt_weight_names:
    #         if 'gesture_lstm_rnn' in weight_name:
    #             new_weight_name = ".".join(weight_name.split(".")[:-1]) + ".0." + weight_name.split(".")[-1]
    #             checkpoint_dict['state_dict'][new_weight_name] = checkpoint_dict['state_dict'].pop(weight_name)
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    d_learning_rate = None
    if d_optimizer is not None:
        d_optimizer.load_state_dict(checkpoint_dict['d_optimizer'])
        d_learning_rate = checkpoint_dict['d_learning_rate']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration, d_optimizer, d_learning_rate


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, d_optimizer=None, d_learning_rate=None):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    if d_optimizer is not None and d_learning_rate is not None:
        torch.save({'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate,
                    'd_optimizer': d_optimizer.state_dict(),
                    'd_learning_rate': d_learning_rate
                    }, filepath,)
    else:
        torch.save({'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, gesture_out):
    """
    Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        # val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=0,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        mel_loss_avg = 0.0
        gesture_loss_avg = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            mel_loss, gesture_loss, g_loss, d_loss = model.subsample_gesture_loss(y_pred, y, criterion)
            # if hparams.freeze_tt2:
            #     reg_loss = gesture_loss
            # else:
            #     reg_loss = mel_loss + gesture_loss
            if distributed_run:
                reduced_mel_loss = reduce_tensor(mel_loss, n_gpus).item()
                reduced_gesture_loss = reduce_tensor(gesture_loss, n_gpus).item()
            else:
                reduced_mel_loss = mel_loss.item()
                reduced_gesture_loss = gesture_loss.item()
            mel_loss_avg += reduced_mel_loss
            gesture_loss_avg += reduced_gesture_loss
        mel_loss_avg = mel_loss_avg / (i + 1)
        gesture_loss_avg = gesture_loss_avg / (i + 1)

    model.train()
    if rank == 0:
        print("Validation mel MSE loss {}: {:9f}  ".format(iteration, mel_loss_avg))
        if gesture_out:
            print("Validation gesture MSE loss {}: {:9f}  ".format(iteration, gesture_loss_avg))
        logger.log_validation(mel_loss_avg, model, y, y_pred, iteration, gesture_out, gesture_loss_avg)

# additional function to freeze part of the model
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

# # get SG_GAN discriminator parameters
def separate_d_params(model):
    d_params = []
    non_d_params = []
    for name, p in model.named_parameters():
        if "SG_d" in name:
            d_params.append(p)
        else:
            non_d_params.append(p)
    return d_params, non_d_params

def get_trainable_params(params, d_params=None):
    params = [x for x in params if x.requires_grad]
    if d_params is not None:
        d_params = [x for x in d_params if x.requires_grad]
    return params, d_params

def train(output_directory, log_directory, warm_start_tt2, n_gpus,
          rank, group_name, hparams, freeze_encoder):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    # checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    d_learning_rate = hparams.d_learning_rate
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # warm start from base tt2
    if warm_start_tt2:
        model, has_new_weights = warm_start_model(
            hparams.tt2_checkpoint, model, hparams.ignore_layers)

    # freeze weights
    if freeze_encoder:
        dfs_freeze(model.encoder)
    if hparams.freeze_tt2:
        print(f"Loading tacotron2 checkpoint: {hparams.tt2_checkpoint}")
        checkpoint_dict = torch.load(hparams.tt2_checkpoint, map_location='cpu')
        tt2_weights = list(checkpoint_dict['state_dict'].keys())
        for name, p in model.named_parameters():
            if name in tt2_weights:
                p.requires_grad = False

    # get optimizer(s)
    d_optimizer = None
    d_params, params = separate_d_params(model)
    if hparams.SG_gan and hparams.separate_d_optimizer:
        d_optimizer = torch.optim.Adam(d_params, lr=d_learning_rate,
                                       weight_decay=hparams.weight_decay)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=learning_rate,
                                     weight_decay=hparams.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                     weight_decay=hparams.weight_decay)

    iteration = 0
    epoch_offset = 0
    # Load checkpoint if one exists
    if hparams.saved_checkpoint is not None:
        model, optimizer, _learning_rate, iteration, d_optimizer, _d_learning_rate = load_checkpoint(
            hparams.saved_checkpoint, model, optimizer, d_optimizer)
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
            d_learning_rate = _d_learning_rate
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    # pdb.set_trace()
    model.train()
    is_overflow = False
    native_amp_scaler = None
    d_native_amp_scaler = None
    if hparams.native_fp16:
        assert not hparams.fp16_run
        native_amp_scaler = torch.cuda.amp.GradScaler()
        if hparams.SG_gan and hparams.separate_d_optimizer:
            d_native_amp_scaler = torch.cuda.amp.GradScaler()
    # train_params, train_d_params = get_trainable_params(params, d_params)

    # mixed-precision
    if hparams.fp16_run:
        assert not hparams.separate_d_optimizer and d_optimizer is None, "not supported with two optimizers"
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O1')

    # distributed
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        if hparams.gesture_use_scheduled_sampling:
            model.decoder.update_p_teacher_forcing(epoch, hparams.gesture_scheduled_sampling_epochs,
                                                   hparams.gesture_scheduled_sampling_min,
                                                   hparams.gesture_scheduled_sampling_max)
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            if d_optimizer is not None:
                for param_group in d_optimizer.param_groups:
                    param_group['lr'] = d_learning_rate

            model.zero_grad()
            # if hparams.SG_gan:
            #     model.SG_d.zero_grad()
            x, y = model.parse_batch(batch)
            if hparams.native_fp16:
                with torch.cuda.amp.autocast():
                    y_pred = model(x)
            else:
                y_pred = model(x)

            if hparams.gesture_out:
                if hparams.native_fp16:
                    with torch.cuda.amp.autocast():
                        mel_loss, gesture_loss, g_loss, d_loss = model.subsample_gesture_loss(y_pred, y, criterion)
                        if hparams.freeze_tt2:
                            tt2_loss = gesture_loss
                        else:
                            tt2_loss = mel_loss + gesture_loss
                        if hparams.SG_gan:
                            tt2_loss += g_loss * hparams.SG_gan_loss_w
                else:
                    mel_loss, gesture_loss, g_loss, d_loss = model.subsample_gesture_loss(y_pred, y, criterion)
                    if hparams.freeze_tt2:
                        tt2_loss = gesture_loss
                    else:
                        tt2_loss = mel_loss + gesture_loss
                    if hparams.SG_gan:
                        tt2_loss += g_loss * hparams.SG_gan_loss_w
            else:
                if hparams.native_fp16:
                    with torch.cuda.amp.autocast():
                        tt2_loss = criterion(y_pred, y)
                else:
                    tt2_loss = criterion(y_pred, y)

            if hparams.distributed_run:
                tt2_reduced_loss = reduce_tensor(tt2_loss, n_gpus)#.item()
                if hparams.SG_gan:
                    d_reduced_loss = reduce_tensor(d_loss, n_gpus)#.item()
            else:
                tt2_reduced_loss = tt2_loss
                if hparams.SG_gan:
                    d_reduced_loss = d_loss

            # if hparams.SG_gan:
            #     g_train_loss = reg_loss + g_loss * hparams.SG_gan_loss_w
            if hparams.fp16_run:
                with amp.scale_loss(tt2_reduced_loss, optimizer) as tt2_scaled_loss:
                    tt2_scaled_loss.backward()
                if hparams.SG_gan:
                    with amp.scale_loss(d_reduced_loss, optimizer) as d_scaled_loss:
                        d_scaled_loss.backward()
            elif hparams.native_fp16:
                native_amp_scaler.scale(tt2_reduced_loss).backward()
                if hparams.SG_gan:
                    d_native_amp_scaler.scale(d_reduced_loss).backward()
            else:
                tt2_reduced_loss.backward()
                if hparams.SG_gan:
                    d_reduced_loss.backward()

            if hparams.clip_grad:
                if hparams.fp16_run:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), hparams.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm)
                elif hparams.native_fp16:
                    # assert False, "not supported"
                    native_amp_scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, hparams.grad_clip_thresh)
                    if hparams.SG_gan and hparams.separate_d_optimizer:
                        d_native_amp_scaler.unsacle_(d_optimizer)
                        d_grad_norm = torch.nn.utils.clip_grad_norm_(
                            d_params, hparams.grad_clip_thresh)
                else:
                    # grad_norm = torch.nn.utils.clip_grad_norm_(
                    #     model.parameters(), hparams.grad_clip_thresh)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, hparams.grad_clip_thresh)
                    if hparams.SG_gan and hparams.separate_d_optimizer:
                        d_grad_norm = torch.nn.utils.clip_grad_norm_(
                            d_params, hparams.grad_clip_thresh)
            else:
                grad_norm = 0

            if hparams.native_fp16:
                native_amp_scaler.step(optimizer)
                native_amp_scaler.update()
                if hparams.SG_gan and hparams.separate_d_optimizer:
                    d_native_amp_scaler.step(d_optimizer)
                    d_native_amp_scaler.update()
            else:
                optimizer.step()
                if hparams.SG_gan and hparams.separate_d_optimizer:
                    d_optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                if hparams.SG_gan and hparams.separate_d_optimizer:
                    print("Train loss {} {:.6f} Grad Norm {:.6f} D Grad Norm {:.6f} {:.2f}s/it".format(
                        iteration, tt2_reduced_loss, grad_norm, d_grad_norm, duration))
                else:
                    print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                        iteration, tt2_reduced_loss, grad_norm, duration))
                # pdb.set_trace()
                curr_p_teacher_forcing = None
                if hparams.gesture_use_scheduled_sampling:
                    curr_p_teacher_forcing = model.decoder.p_teacher_forcing

                if hparams.gesture_out:
                    logger.log_training(
                        tt2_reduced_loss, grad_norm, learning_rate, duration, iteration,
                        curr_p_teacher_forcing, mel_loss=mel_loss, gesture_loss=gesture_loss,
                        g_loss=g_loss, d_loss=d_loss, d_learning_rate=d_learning_rate)
                else:
                    logger.log_training(
                        tt2_reduced_loss, grad_norm, learning_rate, duration, iteration,
                        curr_p_teacher_forcing)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                # validate(model, criterion, valset, iteration, # valset audios have no corresponding motion data
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, hparams.gesture_out)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path, d_optimizer, d_learning_rate)

            iteration += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--model_config', type=str,
                        required=False, help='yaml config for model')

    args = parser.parse_args()

    if args.model_config is not None and args.model_config != "":
        model_config = yaml.load(open(args.model_config, 'r'), Loader=yaml.SafeLoader)

        shared_hparams = model_config["shared_hparams"]
        if "train_hparams" in model_config.keys() and model_config["train_hparams"] is not None:
            train_hparams = model_config["train_hparams"]
    else:
        raise ValueError("Must provide valid model config file.")
    hparams = create_hparams(shared_hparams, train_hparams)
    model_name = os.path.splitext(os.path.basename(args.model_config))[0]
    model_directory = f"{hparams.output_directory}/{model_name}"

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    print("Native FP16 Run:", hparams.native_fp16)

    assert not hparams.fp16_run, "not supported."

    train(model_directory, hparams.log_directory,
          hparams.warm_start_tt2, hparams.n_gpus, args.rank, args.group_name, hparams,
          hparams.freeze_encoder)
