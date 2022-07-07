from model import RNN
from load_data import NameDataset, load_data
from train import train

import torch
import torch.nn as nn


def main():
    device = torch.cuda.current_device()
    print(torch.cuda.get_device_name(device))

    # Load the data
    x_data, y_data = load_data()
    dataset = NameDataset(x_data, y_data)

    n_categories = 3
    n_letters = 57

    rnn = RNN(input_dim=n_letters, hidden_dim=2**12, layer_dim=1,
              output_dim=n_categories)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    train(rnn, optimizer, criterion, dataset)


if __name__ == "__main__":
    main()
