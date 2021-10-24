import torch
from torch import nn
import pdb

class LSTM_D(nn.Module):
    def __init__(self, hparams):
        super(LSTM_D, self).__init__()

        self.lstm = nn.LSTM(
            input_size=hparams.n_mel_channels + hparams.gesture_dim,
            hidden_size=hparams.lstm_d_dim,
            num_layers=hparams.lstm_d_l,
            batch_first=True,
            dropout=hparams.p_SG_d_dropout)

        self.lstm_d_use_both_hc = hparams.lstm_d_use_both_hc
        if self.lstm_d_use_both_hc:
            self.linear_proj = nn.Linear(in_features=hparams.lstm_d_dim*2, out_features=1)
        else:
            self.linear_proj = nn.Linear(in_features=hparams.lstm_d_dim, out_features=1)
        self.loss_f = nn.BCEWithLogitsLoss()
        self.lstm_d_dim=hparams.lstm_d_dim

    def forward(self, x, len_mask=None):
        """

        :param x: expecting (bs, seq, feat)
        :param len_mask: (bs, 1, seq)
        :return:
        """
        o, (h,c) = self.lstm(x)
        if self.lstm_d_use_both_hc:
            h = h[-1] # last layer, (num_layers, bs, dim)
            c = c[-1]
            x = torch.cat((h,c),dim=-1)
            x = self.linear_proj(x)
        else:
            len_mask_ = len_mask.squeeze(1).clone().detach()
            len_index = torch.sum(len_mask_, dim=-1) - 1
            len_index = len_index.unsqueeze(0).unsqueeze(-1).long()
            len_index = len_index.expand(1,-1,self.lstm_d_dim)
            # pdb.set_trace()
            o = o.gather(dim=1, index=len_index).squeeze(0)
            x = self.linear_proj(o)

        x = x.squeeze(-1)
        return x

    def loss(self, y, label, len_mask):
        """

        :param y:
        :param label:
        :param len_mask: (bs, 1, seq)
        :return:
        """
        assert label == 0 or label == 1
        if label == 0:
            loss = self.loss_f(y, torch.zeros(y.shape).cuda())
        else: # label == 1:
            loss = self.loss_f(y, torch.ones(y.shape).cuda())
        return loss