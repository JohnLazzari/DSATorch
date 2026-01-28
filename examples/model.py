import torch.nn as nn


class Model(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(Model, self).__init__()
        self.rnn = nn.RNN(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, input, hx):
        hx, _ = self.rnn(input, hx)
        out = self.out_layer(hx)
        return out, hx
