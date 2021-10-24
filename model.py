from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
import random
import pdb
import numpy as np
from gan import LSTM_D

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, p_dropout=0.5, no_dropout_at_test=False, use_bias=False,
                 no_relu_last_layer=False): # both dropout params default as original tacotron2
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=use_bias)
             for (in_size, out_size) in zip(in_sizes, sizes)])

        self.p_dropout = p_dropout
        self.no_dropout_at_test = no_dropout_at_test
        self.no_relu_last_layer = no_relu_last_layer

    def forward(self, x):
        for l_i, linear in enumerate(self.layers):
            if l_i == len(self.layers) - 1 and self.no_relu_last_layer:
                x = linear(x)
                return x

            # x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
            if self.no_dropout_at_test:
                x = F.dropout(F.relu(linear(x)), p=self.p_dropout, training=self.training)
            else:
                x = F.dropout(F.relu(linear(x)), p=self.p_dropout, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams, type="mel"): # types: mel, gesture
        if type == "mel":
            in_dim = hparams.n_mel_channels
            embed_dim = hparams.postnet_embedding_dim
            n_convs = hparams.postnet_n_convolutions
            kernel_size = hparams.postnet_kernel_size

        elif type == "gesture":
            in_dim = hparams.gesture_dim
            embed_dim = hparams.postnet_embedding_dim
            n_convs = hparams.postnet_n_convolutions
            kernel_size = hparams.postnet_kernel_size

        else:
            assert False

        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(in_dim, embed_dim,
                         kernel_size=kernel_size, stride=1,
                         padding=int((kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(embed_dim))
        )

        for i in range(1, n_convs - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(embed_dim,
                             embed_dim,
                             kernel_size=kernel_size, stride=1,
                             padding=int((kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(embed_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(embed_dim, in_dim,
                         kernel_size=kernel_size, stride=1,
                         padding=int((kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(in_dim))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        if hparams.speakers:
            self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.speaker_embedding_dim
        else:
            self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
                hparams.prenet_dim + self.encoder_embedding_dim,
                hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
                hparams.attention_rnn_dim + self.encoder_embedding_dim,
                hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
                hparams.decoder_rnn_dim + self.encoder_embedding_dim,
                hparams.n_mel_channels * hparams.n_frames_per_step)

        # attn rnn to gesture
        self.attn_to_gesture_dim = hparams.attention_rnn_dim + self.encoder_embedding_dim
        self.attn_to_gesture_use_mlp = hparams.attn_to_gesture_use_mlp
        if self.attn_to_gesture_use_mlp:
            self.attn_to_gesture_mlp = Prenet(in_dim=self.attn_to_gesture_dim,
                                              sizes=[hparams.attn_to_gesture_mlp_dim for _ in
                                                     range(hparams.attn_to_gesture_mlp_l)],
                                              p_dropout=hparams.p_dropout_attn_to_gesture_mlp,
                                              no_dropout_at_test=True,
                                              use_bias=True,
                                              no_relu_last_layer=True)
            self.attn_to_gesture_dim = hparams.attn_to_gesture_mlp_dim

        # gesture module
        self.gesture_out = hparams.gesture_out
        self.gesture_dim = hparams.gesture_dim
        self.gesture_lstm = hparams.gesture_lstm
        self.gesture_to_prenet = hparams.gesture_to_prenet
        self.gesture_lstm_use_prenet = hparams.gesture_lstm_use_prenet
        if self.gesture_out:
            assert self.gesture_dim > 0
            if not self.gesture_lstm: # no separate gesture lstm/share last lstm with speech
                self.linear_projection_gesture = LinearNorm(
                    self.attn_to_gesture_dim,
                    self.gesture_dim * hparams.n_frames_per_step)
            elif self.gesture_lstm: # separate gesture lstm
                self.gesture_lstm_dim = hparams.gesture_lstm_dim
                self.p_gesture_lstm_dropout = hparams.p_gesture_lstm_dropout

                prior_gesture_dim = hparams.gesture_dim
                if self.gesture_lstm_use_prenet:
                    self.gesture_lstm_prenet = Prenet(
                        hparams.gesture_dim * hparams.n_frames_per_step,
                        [hparams.gesture_prenet_dim, hparams.gesture_prenet_dim],
                        p_dropout=hparams.p_gesture_prenet_dropout,
                        no_dropout_at_test=hparams.gesture_prenet_no_dropout_at_test)
                    prior_gesture_dim = hparams.gesture_prenet_dim

                self.gesture_lstm_rnn = [nn.LSTMCell(
                    prior_gesture_dim + self.attn_to_gesture_dim, #hparams.attention_rnn_dim + self.encoder_embedding_dim,
                    hparams.gesture_lstm_dim, 1)]
                for layer_i in range(hparams.gesture_lstm_l-1):
                    self.gesture_lstm_rnn += [nn.LSTMCell(
                        hparams.gesture_lstm_dim,
                        hparams.gesture_lstm_dim, 1)]
                self.gesture_lstm_rnn = nn.ModuleList(self.gesture_lstm_rnn)
                self.gesture_lstm_l = hparams.gesture_lstm_l

                self.encoder_embedding_to_gesture_lstm = hparams.encoder_embedding_to_gesture_lstm
                if self.encoder_embedding_to_gesture_lstm:
                    self.linear_projection_gesture = LinearNorm(
                        hparams.gesture_lstm_dim + self.encoder_embedding_dim,
                        self.gesture_dim * hparams.n_frames_per_step)
                else:
                    self.linear_projection_gesture = LinearNorm(
                        hparams.gesture_lstm_dim,
                        self.gesture_dim * hparams.n_frames_per_step)

            if self.gesture_to_prenet:
                assert self.gesture_dim > 0 and self.gesture_out
                self.prenet = Prenet(
                    (hparams.n_mel_channels + hparams.gesture_dim) * hparams.n_frames_per_step,
                    [hparams.prenet_dim, hparams.prenet_dim])
        self.p_teacher_forcing = 1.0
        self.gesture_use_scheduled_sampling = hparams.gesture_use_scheduled_sampling
        self.gesture_lstm_subsample = hparams.gesture_lstm_subsample
        self.gesture_lstm_step = True # first step is an out step
        self.prior_gesture_lstm_out = None

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def update_p_teacher_forcing(self, curr_epoch, scheduled_sampling_epochs, min_p, max_p=1.0):
        if curr_epoch <= scheduled_sampling_epochs[0]:
            self.p_teacher_forcing = max_p
        elif curr_epoch <= scheduled_sampling_epochs[0] + scheduled_sampling_epochs[1]:
            self.p_teacher_forcing = max_p - \
                                     float(curr_epoch - scheduled_sampling_epochs[0]) / scheduled_sampling_epochs[1] * \
                                     (max_p - min_p)
        else:
            self.p_teacher_forcing = min_p

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        if self.gesture_to_prenet:
            decoder_input = Variable(memory.data.new(
                B, (self.n_mel_channels + self.gesture_dim)* self.n_frames_per_step).zero_())
        else:
            decoder_input = Variable(memory.data.new(
                B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def get_go_frame_gesture(self, memory):
        """ Gets all zeros frames to use as first input to gesture prenet/lstm
        """
        B = memory.size(0)
        gesture_go_frame = Variable(memory.data.new(
                B, self.gesture_dim * self.n_frames_per_step).zero_())
        return gesture_go_frame

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        if self.gesture_lstm:
            self.gesture_lstm_hidden = [Variable(memory.data.new(
                B, self.gesture_lstm_dim).zero_()) for _ in range(self.gesture_lstm_l)]
            self.gesture_lstm_cell = [Variable(memory.data.new(
                B, self.gesture_lstm_dim).zero_()) for _ in range(self.gesture_lstm_l)]

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        if not self.gesture_out:
            mel_outputs = mel_outputs.view(
                mel_outputs.size(0), -1, self.n_mel_channels)
        else:
            mel_outputs = mel_outputs.view(
                mel_outputs.size(0), -1, self.n_mel_channels + self.gesture_dim)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, prior_gesture_to_glstm=None):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context_to_linear = torch.cat(
                (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context_to_linear)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        if self.gesture_out:
            if not self.gesture_lstm:
                decoder_output_gesture = self.linear_projection_gesture(
                    decoder_hidden_attention_context_to_linear)
            elif self.gesture_lstm and not self.gesture_lstm_step:  # not an out step, copy prior out
                decoder_output_gesture = self.prior_gesutre_lstm_out
            else:
                assert not type(prior_gesture_to_glstm) == type(None)

                self.gesture_lstm_hidden[0], self.gesture_lstm_cell[0] = self.gesture_lstm_rnn[0](
                    torch.cat((decoder_input, prior_gesture_to_glstm), -1),
                    (self.gesture_lstm_hidden[0], self.gesture_lstm_cell[0]))
                self.gesture_lstm_hidden[0] = F.dropout(
                    self.gesture_lstm_hidden[0], self.p_gesture_lstm_dropout, self.training)
                for layer_i in range(1, self.gesture_lstm_l):
                    self.gesture_lstm_hidden[layer_i], self.gesture_lstm_cell[layer_i] = self.gesture_lstm_rnn[layer_i](
                        self.gesture_lstm_hidden[layer_i-1],
                        (self.gesture_lstm_hidden[layer_i], self.gesture_lstm_cell[layer_i]))
                    self.gesture_lstm_hidden[layer_i] = F.dropout(
                        self.gesture_lstm_hidden[layer_i], self.p_gesture_lstm_dropout, self.training)
                if self.encoder_embedding_to_gesture_lstm:
                    gesture_hidden_attention_context_to_linear = torch.cat(
                        (self.gesture_lstm_hidden[-1], self.attention_context), dim=1)
                else:
                    gesture_hidden_attention_context_to_linear = self.gesture_lstm_hidden[-1]

                decoder_output_gesture = self.linear_projection_gesture(gesture_hidden_attention_context_to_linear)
                self.prior_gesutre_lstm_out = decoder_output_gesture

            decoder_output = torch.cat((decoder_output, decoder_output_gesture), axis=1)
            return decoder_output, gate_prediction, self.attention_weights

        else:
            return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        if self.gesture_lstm:
            mels, gesture_arrs = decoder_inputs[:,:self.n_mel_channels], decoder_inputs[:,self.n_mel_channels:]

            gesture_go_frame = self.get_go_frame_gesture(memory).unsqueeze(0)
            gesture_frames = self.parse_decoder_inputs(gesture_arrs)
            gesture_frames = torch.cat((gesture_go_frame, gesture_frames), dim=0)
            if self.gesture_lstm_use_prenet:
                gesture_frames = self.gesture_lstm_prenet(gesture_frames)

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        if not self.gesture_to_prenet:
            mels = decoder_inputs[:,:self.n_mel_channels]
            decoder_inputs = mels
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        decoder_outputs, gate_outputs, alignments = [], [], []
        while len(decoder_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(decoder_outputs)]
            prior_gesture_to_glstm = None
            if self.gesture_lstm and len(decoder_outputs) % self.gesture_lstm_subsample == 0:
                self.gesture_lstm_step = True
                if len(decoder_outputs) == 0:
                    prior_gesture_idx_teacher_forcing = 0
                else:
                    prior_gesture_idx_teacher_forcing = len(decoder_outputs) - self.gesture_lstm_subsample + 1
                if self.gesture_use_scheduled_sampling:
                    sampled_float = random.random()

                    if sampled_float <= self.p_teacher_forcing and self.p_teacher_forcing != 0.0:# teacher forcing
                        prior_gesture_to_glstm = gesture_frames[prior_gesture_idx_teacher_forcing]
                    else: # auto-regressive
                        if len(decoder_outputs) > 0:
                            if self.gesture_lstm_use_prenet:
                                prior_gesture_to_glstm = self.gesture_lstm_prenet(self.prior_gesutre_lstm_out)
                            else:
                                prior_gesture_to_glstm = self.prior_gesutre_lstm_out
                        else: # first frame
                            assert len(decoder_outputs) == 0
                            prior_gesture_to_glstm = gesture_frames[0]
                else:
                    prior_gesture_to_glstm = gesture_frames[prior_gesture_idx_teacher_forcing]

            elif self.gesture_lstm: # not a gesture lstm out step due to subsample
                self.gesture_lstm_step = False

            decoder_output, gate_output, attention_weights = self.decode(
                    decoder_input, prior_gesture_to_glstm=prior_gesture_to_glstm)

            decoder_outputs += [decoder_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        decoder_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            decoder_outputs, gate_outputs, alignments)
        return decoder_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)
        if self.gesture_lstm:
            self.prior_gesutre_lstm_out = self.get_go_frame_gesture(memory)

        self.initialize_decoder_states(memory, mask=None)

        decoder_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            prior_gesture_to_glstm = None
            if self.gesture_lstm:
                if len(decoder_outputs) % self.gesture_lstm_subsample == 0:
                    self.gesture_lstm_step = True
                    if self.gesture_lstm_use_prenet:
                        prior_gesture_to_glstm = self.gesture_lstm_prenet(self.prior_gesutre_lstm_out)
                    else:
                        prior_gesture_to_glstm = self.prior_gesutre_lstm_out
                else:
                    self.gesture_lstm_step = False

            decoder_output, gate_output, alignment = self.decode(decoder_input,
                    prior_gesture_to_glstm=prior_gesture_to_glstm)

            decoder_outputs += [decoder_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(decoder_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            if self.gesture_to_prenet:
                decoder_input = decoder_output
            else:
                decoder_input = decoder_output[:,:self.n_mel_channels]

        decoder_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            decoder_outputs, gate_outputs, alignments)

        return decoder_outputs, gate_outputs, alignments

class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.speakers = hparams.speakers
        if self.speakers:
            self.speaker_embedding = nn.Embedding(
                hparams.n_speakers, hparams.speaker_embedding_dim)
        self.gesture_out = hparams.gesture_out
        self.gesture_dim = hparams.gesture_dim
        self.gesture_use_postnet = hparams.gesture_use_postnet
        if self.gesture_out and self.gesture_use_postnet:
            self.postnet_gesture = Postnet(hparams, type="gesture")
        self.gesture_to_prenet = hparams.gesture_to_prenet
        if self.gesture_to_prenet:
            assert self.gesture_dim > 0 and self.gesture_out
        self.gesture_lstm = hparams.gesture_lstm
        self.gesture_lstm_subsample = hparams.gesture_lstm_subsample
        self.gesture_use_moglow = hparams.gesture_use_moglow

        self.SG_gan = hparams.SG_gan
        if self.SG_gan:
            self.SG_d = LSTM_D(hparams)
            assert self.gesture_out, "SG_gan can only be used when gesture_out==True"

        self.glstm_at_mel = hparams.glstm_at_mel
        if self.glstm_at_mel:
            assert not self.gesture_lstm

            self.gesture_lstm_rnn = [nn.LSTMCell(
                self.gesture_dim + self.n_mel_channels,
                hparams.gesture_lstm_dim, 1)]
            for layer_i in range(hparams.gesture_lstm_l - 1):
                self.gesture_lstm_rnn += [nn.LSTMCell(
                    hparams.gesture_lstm_dim,
                    hparams.gesture_lstm_dim, 1)]
            self.gesture_lstm_rnn = nn.ModuleList(self.gesture_lstm_rnn)

            self.linear_projection_gesture = LinearNorm(
                hparams.gesture_lstm_dim,
                self.gesture_dim * hparams.n_frames_per_step)

            self.gesture_lstm_dim = hparams.gesture_lstm_dim
            self.gesture_lstm_l = hparams.gesture_lstm_l
            self.p_gesture_lstm_dropout = hparams.p_gesture_lstm_dropout

        self.add_misalign = hparams.add_misalign
        self.misalign_min = hparams.misalign_min
        self.misalign_max = hparams.misalign_max

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_feats, audio_paths, gesture_arrs = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speaker_feats = to_gpu(speaker_feats).long()

        out_arr = mel_padded
        if self.gesture_out:
            assert gesture_arrs.shape[1] == self.gesture_dim
            gesture_arrs = to_gpu(gesture_arrs).float()
            out_arr = torch.cat((out_arr, gesture_arrs), dim=1)

        return (
            (text_padded, input_lengths, mel_padded, max_len,
             output_lengths, speaker_feats, gesture_arrs),
            (out_arr, gate_padded))

    def parse_batch_for_eval(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, speaker_feats, audio_paths, gesture_arrs = batch
        # pdb.set_trace()
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.cuda().float()
        gate_padded = gate_padded.cuda().float()
        output_lengths = output_lengths.cuda().long()
        speaker_feats = to_gpu(speaker_feats).long()

        out_arr = mel_padded
        if self.gesture_out:
            assert gesture_arrs.shape[1] == self.gesture_dim
            gesture_arrs = gesture_arrs.cuda().float()
            out_arr = torch.cat((out_arr, gesture_arrs), dim=1)

        return (
            (text_padded, input_lengths, mel_padded, max_len,
             output_lengths, speaker_feats, None),
            (out_arr, gate_padded))
    input_text_i_in_eval_batch = 0
    speaker_feats_i_in_eval_batch = -2

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            if self.gesture_out and (not self.gesture_use_moglow or not self.training):
                mask = mask.expand(self.n_mel_channels+self.gesture_dim, mask.size(0), mask.size(1))
            else:
                mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, \
            output_lengths, speaker_feats, gesture_arrs = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        embedded_text = self.encoder(embedded_inputs, text_lengths)

        if self.speakers:
            embedded_speakers = self.speaker_embedding(speaker_feats)[:, None]
            embedded_feats = embedded_speakers.repeat(1, embedded_text.size(1), 1)

            encoder_outputs = torch.cat(
                (embedded_text, embedded_feats), dim=2)
        else:
            encoder_outputs = embedded_text

        if self.gesture_to_prenet or self.gesture_lstm:
            decoder_outputs, gate_outputs, alignments = self.decoder(
                encoder_outputs, torch.cat((mels, gesture_arrs), dim=1), memory_lengths=text_lengths)
        else:
            decoder_outputs, gate_outputs, alignments = self.decoder(
                encoder_outputs, mels, memory_lengths=text_lengths)

        if self.gesture_out:
            assert decoder_outputs.shape[1] == self.n_mel_channels + self.gesture_dim
            mel_outputs, gesture_outputs = decoder_outputs[:,:-self.gesture_dim], decoder_outputs[:,-self.gesture_dim:]

            mel_outputs_postnet = self.postnet(mel_outputs)
            mel_outputs_postnet = mel_outputs + mel_outputs_postnet

            if self.gesture_use_postnet:
                gesture_outputs_postnet = self.postnet_gesture(gesture_outputs)
                gesture_outputs_postnet = gesture_outputs + gesture_outputs_postnet

                outputs = torch.cat((mel_outputs_postnet, gesture_outputs_postnet), axis=1)

            elif self.glstm_at_mel:
                B, _, mel_steps = mel_outputs_postnet.shape

                gesture_go_frame = Variable(mel_outputs_postnet.data.new(
                    B, self.gesture_dim).zero_())
                gesture_lstm_hidden = [Variable(mel_outputs_postnet.data.new(
                    B, self.gesture_lstm_dim).zero_()) for _ in range(self.gesture_lstm_l)]
                gesture_lstm_cell = [Variable(mel_outputs_postnet.data.new(
                    B, self.gesture_lstm_dim).zero_()) for _ in range(self.gesture_lstm_l)]
                prior_gesture_to_glstm = gesture_go_frame
                gesture_outputs = []
                for mel_step_i in range(mel_steps):
                    if mel_step_i % self.gesture_lstm_subsample == 0:
                        prior_gesture_idx_teacher_forcing = len(gesture_outputs) - self.gesture_lstm_subsample
                        if self.decoder.gesture_use_scheduled_sampling:
                            sampled_float = random.random()

                            if sampled_float <= self.decoder.p_teacher_forcing and self.decoder.p_teacher_forcing != 0.0 \
                                    and len(gesture_outputs) != 0:  # teacher forcing
                                prior_gesture_to_glstm = gesture_arrs[:,:,prior_gesture_idx_teacher_forcing]


                        gesture_lstm_hidden[0], gesture_lstm_cell[0] = self.gesture_lstm_rnn[0](
                            torch.cat((mel_outputs_postnet[:,:,mel_step_i], prior_gesture_to_glstm), -1),
                            (gesture_lstm_hidden[0], gesture_lstm_cell[0]))
                        gesture_lstm_hidden[0] = F.dropout(
                            gesture_lstm_hidden[0], self.p_gesture_lstm_dropout, self.training)
                        for layer_i in range(1, self.gesture_lstm_l):
                            gesture_lstm_hidden[layer_i], gesture_lstm_cell[layer_i] = self.gesture_lstm_rnn[layer_i](
                                gesture_lstm_hidden[layer_i - 1],
                                (gesture_lstm_hidden[layer_i], gesture_lstm_cell[layer_i]))
                            gesture_lstm_hidden[layer_i] = F.dropout(
                                gesture_lstm_hidden[layer_i], self.p_gesture_lstm_dropout, self.training)
                        prior_gesture_to_glstm = self.linear_projection_gesture(gesture_lstm_hidden[-1])
                    gesture_outputs.append(prior_gesture_to_glstm)

                gesture_outputs = torch.stack(gesture_outputs).transpose(0, 1).transpose(1,2).contiguous()
                outputs = torch.cat((mel_outputs_postnet, gesture_outputs), axis=1)

            else:
                outputs = torch.cat((mel_outputs_postnet, gesture_outputs), axis=1)

        else:
            mel_outputs_postnet = self.postnet(decoder_outputs)
            outputs = decoder_outputs + mel_outputs_postnet

        return self.parse_output(
            [decoder_outputs, outputs, gate_outputs, alignments],
            output_lengths)

    # assert batch_size == 1
    def inference(self, inputs, speaker_feat, mels=None):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        if self.speakers:
            embedded_speakers = self.speaker_embedding(speaker_feat)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat(
                (encoder_outputs, embedded_speakers), dim=2)

        decoder_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        if self.gesture_out:
            assert decoder_outputs.shape[1] == self.n_mel_channels + self.gesture_dim
            mel_outputs, gesture_outputs = decoder_outputs[:,:-self.gesture_dim], decoder_outputs[:,-self.gesture_dim:]

            mel_outputs_postnet = self.postnet(mel_outputs)
            mel_outputs_postnet = mel_outputs + mel_outputs_postnet

            if self.gesture_use_postnet:
                gesture_outputs_postnet = self.postnet_gesture(gesture_outputs)
                gesture_outputs_postnet = gesture_outputs + gesture_outputs_postnet

                outputs = torch.cat((mel_outputs_postnet, gesture_outputs_postnet), axis=1)

            elif self.glstm_at_mel:
                B, _, mel_steps = mel_outputs_postnet.shape

                gesture_go_frame = Variable(mel_outputs_postnet.data.new(
                    B, self.gesture_dim).zero_())
                gesture_lstm_hidden = [Variable(mel_outputs_postnet.data.new(
                    B, self.gesture_lstm_dim).zero_()) for _ in range(self.gesture_lstm_l)]
                gesture_lstm_cell = [Variable(mel_outputs_postnet.data.new(
                    B, self.gesture_lstm_dim).zero_()) for _ in range(self.gesture_lstm_l)]
                prior_gesture_to_glstm = gesture_go_frame
                gesture_outputs = []
                for mel_step_i in range(mel_steps):
                    if mel_step_i % self.gesture_lstm_subsample == 0:
                        gesture_lstm_hidden[0], gesture_lstm_cell[0] = self.gesture_lstm_rnn[0](
                            torch.cat((mel_outputs_postnet[:,:,mel_step_i], prior_gesture_to_glstm), -1),
                            (gesture_lstm_hidden[0], gesture_lstm_cell[0]))
                        gesture_lstm_hidden[0] = F.dropout(
                            gesture_lstm_hidden[0], self.p_gesture_lstm_dropout, self.training)
                        for layer_i in range(1, self.gesture_lstm_l):
                            gesture_lstm_hidden[layer_i], gesture_lstm_cell[layer_i] = self.gesture_lstm_rnn[layer_i](
                                gesture_lstm_hidden[layer_i - 1],
                                (gesture_lstm_hidden[layer_i], gesture_lstm_cell[layer_i]))
                            gesture_lstm_hidden[layer_i] = F.dropout(
                                gesture_lstm_hidden[layer_i], self.p_gesture_lstm_dropout, self.training)
                        prior_gesture_to_glstm = self.linear_projection_gesture(gesture_lstm_hidden[-1])
                    gesture_outputs.append(prior_gesture_to_glstm)

                gesture_outputs = torch.stack(gesture_outputs).transpose(0, 1).transpose(1,2).contiguous()
                outputs = torch.cat((mel_outputs_postnet, gesture_outputs), axis=1)

            else:
                outputs = torch.cat((mel_outputs_postnet, gesture_outputs), axis=1)
        else:
            mel_outputs_postnet = self.postnet(decoder_outputs)
            outputs = decoder_outputs + mel_outputs_postnet

        return self.parse_output(
            [decoder_outputs, outputs, gate_outputs, alignments])
    pre_postnet_out_i = 0
    postnet_out_i = 1

    def misalign_mel_gesture(self, dat):
        """
        shifting gesture dims to create misalignment
        :param dat: (bs, feats, l), feats = self.n_mel_channels + self.gesture_dim
        :return:
        """
        ma = self.misalign_min + (self.misalign_max - self.misalign_min) * random.random()
        ma = int(ma)

        if random.random() > 0.5: # shifting left
            dat_left = dat[:, self.n_mel_channels:, :ma]
            dat[:, self.n_mel_channels:, :-ma] = dat[:, self.n_mel_channels:, ma:]
            dat[:, self.n_mel_channels:, -ma:] = dat_left
        else: # shifting right
            dat_right = dat[:, self.n_mel_channels:, -ma:]
            dat[:, self.n_mel_channels:, ma:] = dat[:, self.n_mel_channels:, :-ma]
            dat[:, self.n_mel_channels:, :ma] = dat_right

        return dat

    def subsample_gesture_loss(self, y_pred, y, tacotron_loss_fn):
        mel_gesture_pred, mel_gesture_pred_postnet, gate_pred, _ = y_pred
        mel_gesture_gt, gate_gt = y[0], y[1]

        mel_pred = mel_gesture_pred[:,:self.n_mel_channels,:]
        gesture_pred = mel_gesture_pred[:,self.n_mel_channels:,:]
        gesture_pred = gesture_pred[:,:,::self.gesture_lstm_subsample]

        mel_pred_postnet = mel_gesture_pred_postnet[:,:self.n_mel_channels,:]
        gesture_pred_postnet = mel_gesture_pred_postnet[:,self.n_mel_channels:,:]
        gesture_pred_postnet = gesture_pred_postnet[:,:,::self.gesture_lstm_subsample]

        mel_gt = mel_gesture_gt[:,:self.n_mel_channels,:]
        gesture_gt = mel_gesture_gt[:,self.n_mel_channels:,:]
        gesture_gt = gesture_gt[:,:,::self.gesture_lstm_subsample]
        subsampled_gt_mask = 1 - gate_gt[:,::self.gesture_lstm_subsample]
        subsampled_gt_mask = subsampled_gt_mask.unsqueeze(1)

        loss_mel = tacotron_loss_fn((mel_pred, mel_pred_postnet, gate_pred, y_pred[-1]),
                                    (mel_gt, gate_gt))

        # always use gesture_postnet
        # if no postnet for gesture, then gesture_pred == gesture_pred_postnet
        loss_gesture = nn.MSELoss()(gesture_gt * subsampled_gt_mask,
                                    gesture_pred_postnet * subsampled_gt_mask)

        g_loss, d_loss = None, None
        if self.SG_gan:
            mel_gesture_pred_postnet_ss = mel_gesture_pred_postnet[:,:,::self.gesture_lstm_subsample]
            mel_gesture_gt_ss = mel_gesture_gt[:,:,::self.gesture_lstm_subsample]

            gen_d = torch.transpose(mel_gesture_pred_postnet_ss,1,2)
            gen_d = self.SG_d(gen_d, subsampled_gt_mask)
            g_loss = self.SG_d.loss(gen_d, 1, subsampled_gt_mask)

            gen_dd = torch.transpose(mel_gesture_pred_postnet_ss,1,2).clone().detach()
            gen_dd = self.SG_d(gen_dd, subsampled_gt_mask)
            gt_dd = torch.transpose(mel_gesture_gt_ss,1,2).clone().detach()
            gt_dd = self.SG_d(gt_dd,subsampled_gt_mask)

            if self.add_misalign:
                gt_ma_dd = torch.transpose(self.misalign_mel_gesture(mel_gesture_gt_ss.clone()),1,2).detach()
                gt_ma_dd = self.SG_d(gt_ma_dd, subsampled_gt_mask)

                d_loss = self.SG_d.loss(gen_dd, 0, subsampled_gt_mask) \
                       + self.SG_d.loss(gt_dd, 1, subsampled_gt_mask) \
                       + self.SG_d.loss(gt_ma_dd, 0, subsampled_gt_mask)

            else:
                d_loss = self.SG_d.loss(gen_dd, 0, subsampled_gt_mask) \
                    + self.SG_d.loss(gt_dd, 1, subsampled_gt_mask)

        return loss_mel, loss_gesture, g_loss, d_loss