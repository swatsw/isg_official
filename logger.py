import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration, p_gesture_scheduled_sampling=None,
                     mel_loss=None, gesture_loss=None,
                     g_loss=None, d_loss=None, d_learning_rate=None):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

        if p_gesture_scheduled_sampling is not None:
            self.add_scalar("p_gesture_scheduled_sampling", p_gesture_scheduled_sampling, iteration)
        if mel_loss is not None:
            self.add_scalar("training.mel_loss", mel_loss, iteration)
        if gesture_loss is not None:
            self.add_scalar("training.gesture_loss", gesture_loss, iteration)
        if g_loss is not None:
            self.add_scalar("training.gan.g_loss", g_loss, iteration)
        if d_loss is not None:
            self.add_scalar("training.gan.d_loss", d_loss, iteration)
        if d_learning_rate is not None:
            self.add_scalar("learning.d_lr", d_learning_rate, iteration)

    def log_validation(self, mel_loss, model, y, y_pred, iteration, has_gesture_loss=False, gesture_loss=0.0):
        self.add_scalar("validation.mel_MSE", mel_loss, iteration)
        if has_gesture_loss:
            self.add_scalar("validation.gesture_MSE", gesture_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
