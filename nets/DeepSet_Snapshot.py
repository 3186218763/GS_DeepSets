import torch
import torch.nn as nn

from nets.DeepSets import DeepSetModel


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super(SAB, self).__init__()
        self.mab = nn.MultiheadAttention(dim_out, num_heads, batch_first=True)
        self.fc_q = nn.Linear(dim_in, dim_out)
        self.fc_k = nn.Linear(dim_in, dim_out)
        self.fc_v = nn.Linear(dim_in, dim_out)

    def forward(self, X):
        Q = self.fc_q(X)
        K, V = self.fc_k(X), self.fc_v(X)
        out, _ = self.mab(Q, K, V)
        return out


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, X, src_key_padding_mask=None):
        Q = self.S.repeat(X.size(0), 1, 1)
        out, _ = self.mab(Q, X, X)
        return out


class Net_Snapshot(nn.Module):
    def __init__(self, dim_input, dim_output, num_outputs=1, dim_hidden=64, num_heads=4):
        super(Net_Snapshot, self).__init__()
        self.feat_in = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=num_heads, dim_feedforward=2 * dim_hidden,
                                                   dropout=0.0, batch_first=True)
        decoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=num_heads, dim_feedforward=2 * dim_hidden,
                                                   dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = PMA(dim_hidden, num_heads, num_outputs)
        self.dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.feat_out = nn.Sequential(
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, x):
        x = self.feat_in(x)
        x = self.enc(x)
        x = self.pool(x)
        x = self.dec(x)
        out = self.feat_out(x)
        return out.squeeze(1)  # 输出形状为 (batch_size, 3)


class DeepSet_Snapshot(nn.Module):
    def __init__(self, deepset_hidden_size: int, deepset_out_size: int, input_size: int = 6, Debug=False):
        super(DeepSet_Snapshot, self).__init__()

        self.net = nn.Sequential(
            DeepSetModel(input_size=input_size, hidden_size=deepset_hidden_size,
                         output_size=deepset_out_size, Debug=Debug),
            Net_Snapshot(dim_input=deepset_out_size, dim_output=3, dim_hidden=1024)
        )

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x * pad_mask
        out = self.net(x)
        return out


if __name__ == '__main__':
    size = (64, 32, 6)
    tensor = torch.rand(size, dtype=torch.float32)
    model = DeepSet_Snapshot(input_size=6, deepset_out_size=64, deepset_hidden_size=64)
    out = model(tensor)
    print(out.shape)
