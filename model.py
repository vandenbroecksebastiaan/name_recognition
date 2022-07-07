import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                          nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sm = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(1024)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)\
                  .requires_grad_().cuda().detach()

        out, hn = self.rnn(x, h0)
        out = out[-1, :]

        out = self.fc(out)
        # out = self.sm(out)

        return out
