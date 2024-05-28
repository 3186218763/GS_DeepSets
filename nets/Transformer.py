from torch import nn
import torch
import torch.nn.functional as F


# 自注意力模块, 用于实现自注意力机制
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, nums_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = nn.MultiheadAttention(dim_in, nums_heads)
        self.fc_q = nn.Linear(dim_in, dim_out)
        self.fc_k = nn.Linear(dim_in, dim_out)
        self.fc_v = nn.Linear(dim_in, dim_out)

        def forward(self, x):
            Q = self.fc_q(x)
            K = self.fc_k(x)
            V = self.fc_v(x)
            out, wts = self.mab(Q, K, V)

            return out, wts


# 多头注意力模块, 用于实现具有位置信息的多头注意力机制
class PMA(nn.Module):
    def __init__(self, dim, num_heads, nums_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(nums_seeds, 1, dim))
        nn.init.xavier_uniform_(self.s)
        self.mab = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x, src_key_padding_mask=None):
        Q = self.S
        out, _ = self.mab(Q, x, x, src_key_padding_mask=src_key_padding_mask)
        return out


# 结合了自注意力块和位置多头注意力机制, 包含了编码器部分和解码器部分
class Net_Snapshot(nn.Module):
    def __init__(self, dim_input, dim_output, num_outputs, dim_hidden=64, num_heads=4):
        super(Net_Snapshot, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=4, dim_feedforward=2 * dim_hidden, dropout=0)
        decoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=4, dim_feedforward=2 * dim_hidden, dropout=0)
        self.feat_in = nn.Linear(dim_input, dim_hidden)  # 这里也可以是多层MLP,

        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.pool = PMA(dim_hidden, num_heads, num_outputs)
        self.fet_out = nn.Linear(dim_hidden, dim_hidden)  # 可以是多层MLP,

    def forward(self, x, pad_mask=None):
        x = self.feat_in(x)
        x = self.enc(x, src_key_padding_mask=pad_mask)
        x = self.pool(x, src_key_padding_mask=pad_mask)
        x = self.dec(x)
        out = self.feat_out(x)
        return torch.squeeze(out, dim=0)


