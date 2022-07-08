import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                          nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc2 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))
        self.fc3 = nn.Linear(int(hidden_dim / 4), output_dim)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        out, hn = self.rnn(x)

        out = out[-1, :]

        out = self.fc1(out)
        out = nn.functional.relu(out)

        out = self.fc2(out)
        out = nn.functional.relu(out)

        out = self.fc3(out)
        out = nn.functional.relu(out)
        out = self.sm(out)

        return out
