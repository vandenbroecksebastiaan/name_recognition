import torch.nn as nn


class model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()

        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=1028,
                           num_layers=3, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(2056, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_features=2048)
        self.batchnorm2 = nn.BatchNorm1d(num_features=1024)
        self.dropout = nn.Dropout(p=0.8)
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