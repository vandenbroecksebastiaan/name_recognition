from model import RNN
from load_data import NameDataset, load_data
from train import train

import torch
import torch.nn as nn


def main():
    torch.cuda.device("cuda")

    # Load the data
    x_data, y_data = load_data()
    dataset = NameDataset(x_data, y_data, batch_size=1)

    n_categories = 3
    n_letters = 57

    rnn = RNN(input_dim=n_letters, hidden_dim=1024, layer_dim=3,
              output_dim=n_categories).cuda()
    train(rnn, dataset)


if __name__ == "__main__":
    main()
