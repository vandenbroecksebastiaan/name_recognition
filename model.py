import torch.nn as nn


class model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, linear_dim, output_dim):
        super(model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, int(linear_dim/2))
        self.fc2 = nn.Linear(int(linear_dim/2), int(linear_dim/4))
        self.fc3 = nn.Linear(int(linear_dim/4), output_dim)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_features=int(linear_dim/2))
        self.batchnorm2 = nn.BatchNorm1d(num_features=int(linear_dim/4))
        self.dropout = nn.Dropout(p=0.7)
        self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        out, hn = self.rnn(x)
        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.fc3(out)