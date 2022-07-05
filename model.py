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
        self.sn = nn.Softmax(dim=0)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim)\
                  .requires_grad_()

        # We can make the x into a "batch" by doing this:
        # x = x.unsqueeze(1)
        x = x.type(torch.float)
        x = torch.squeeze(x, dim=1)

        # The hidden state should also be 2d in this case:
        h0 = h0[0].detach()

        out, hn = self.rnn(x, h0)

        out = out[-1, :]
        out = self.fc(out)
        out = self.sn(out)

        return out
